# src/utils/security.py
import hashlib
import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from src.utils.config import settings


def hash_pdf_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive_session_key(session_id: str) -> bytes:
    secret = settings.session_secret_key.encode()
    h = hashlib.pbkdf2_hmac(
        'sha256',
        session_id.encode(),
        secret,
        iterations=100000,
        dklen=32,
    )
    return h


def encrypt_session_data(session_id: str, plaintext: bytes) -> bytes:
    key = derive_session_key(session_id)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt_session_data(session_id: str, ciphertext_with_nonce: bytes) -> bytes:
    key = derive_session_key(session_id)
    aesgcm = AESGCM(key)
    nonce = ciphertext_with_nonce[:12]
    ciphertext = ciphertext_with_nonce[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)
