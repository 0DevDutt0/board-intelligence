# src/api/session.py
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional
from src.utils.config import settings
from src.utils.security import encrypt_session_data, decrypt_session_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionStore:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._encrypted_states = {}
        self._live_objects = {}
        self._cleanup_task = None

    def start_cleanup(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            await self._expire_old_sessions()

    async def _expire_old_sessions(self):
        now = datetime.utcnow()
        timeout = timedelta(minutes=settings.session_timeout_minutes)
        expired = []
        async with self._lock:
            for sid, obj in self._live_objects.items():
                last_active = obj.get('last_active')
                if last_active and (now - last_active) > timeout:
                    expired.append(sid)
        for sid in expired:
            logger.info(f'Session {sid} expired, cleaning up')
            await self.delete_session(sid)

    async def create_session(self, initial_state: dict) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        state = {
            'session_id': session_id,
            'doc_hash': initial_state.get('doc_hash', ''),
            'doc_filename': initial_state.get('doc_filename', ''),
            'page_count': initial_state.get('page_count', 0),
            'chunk_count': initial_state.get('chunk_count', 0),
            'conversation_history': [],
            'created_at': now.isoformat(),
            'last_active': now,
            'ingest_stats': initial_state.get('ingest_stats', {}),
            'valid_pages': initial_state.get('valid_pages', []),
        }

        plain = json.dumps({
            k: v for k, v in state.items()
            if k != 'last_active'
        }).encode()
        encrypted = encrypt_session_data(session_id, plain)

        async with self._lock:
            self._encrypted_states[session_id] = encrypted
            self._live_objects[session_id] = {
                'last_active': now,
                'pipeline': initial_state.get('pipeline', {}),
            }
        return session_id

    async def get_session(self, session_id: str) -> Optional[dict]:
        async with self._lock:
            if session_id not in self._encrypted_states:
                return None
            encrypted = self._encrypted_states[session_id]
            live = self._live_objects.get(session_id, {})

        plain = decrypt_session_data(session_id, encrypted)
        state = json.loads(plain)
        state['last_active'] = live.get('last_active')
        state['pipeline'] = live.get('pipeline', {})
        return state

    async def update_session(self, session_id: str, updates: dict) -> bool:
        async with self._lock:
            if session_id not in self._encrypted_states:
                return False
            encrypted = self._encrypted_states[session_id]

        plain = decrypt_session_data(session_id, encrypted)
        state = json.loads(plain)

        pipeline_update = updates.pop('pipeline', None)
        for key, val in updates.items():
            state[key] = val

        plain_new = json.dumps(state).encode()
        encrypted_new = encrypt_session_data(session_id, plain_new)

        async with self._lock:
            self._encrypted_states[session_id] = encrypted_new
            self._live_objects[session_id]['last_active'] = datetime.utcnow()
            if pipeline_update is not None:
                self._live_objects[session_id]['pipeline'].update(pipeline_update)
        return True

    async def delete_session(self, session_id: str) -> bool:
        import shutil
        import os
        async with self._lock:
            found = session_id in self._encrypted_states
            self._encrypted_states.pop(session_id, None)
            self._live_objects.pop(session_id, None)

        if found:
            session_dir = settings.temp_session_path / session_id
            if session_dir.exists():
                pdf_path = session_dir / 'document.pdf'
                if pdf_path.exists():
                    size = pdf_path.stat().st_size
                    with open(pdf_path, 'r+b') as f:
                        f.write(b'\x00' * size)
                shutil.rmtree(session_dir, ignore_errors=True)
            logger.info(f'Session {session_id} deleted and temp files cleaned')
        return found

    async def append_conversation(self, session_id: str, user: str, assistant: str):
        state = await self.get_session(session_id)
        if state is None:
            return
        history = state.get('conversation_history', [])
        history.append({'user': user, 'assistant': assistant})
        if len(history) > settings.max_conversation_history:
            history = history[-settings.max_conversation_history:]
        await self.update_session(session_id, {'conversation_history': history})

    async def get_pipeline(self, session_id: str) -> dict:
        async with self._lock:
            live = self._live_objects.get(session_id)
            if live is None:
                return {}
            return live.get('pipeline', {})


session_store = SessionStore()
