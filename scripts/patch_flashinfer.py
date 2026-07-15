# scripts/patch_flashinfer.py
# Patches two flashinfer functions that reject sm_120 (RTX 5090 Blackwell).
# Run inside the WSL vLLM venv after flashinfer is installed; invoked
# automatically by scripts/start_vllm_wsl.sh on first install. Idempotent.
#
# Patch 1 -- flashinfer/jit/core.py :: check_cuda_arch()
#   Stock version uses torch.cuda.get_arch_list() which may exclude sm_120.
#   Replaced with torch.cuda.get_device_capability() direct query.
#
# Patch 2 -- flashinfer/compilation_context.py :: CompilationContext.get_nvcc_flags_list()
#   Internal SM-major -> CUDA-major lookup table does not know about sm_120.
#   A monkey-patch is appended to the file that reads FLASHINFER_CUDA_ARCHS
#   (set to "12.0" in scripts/start_vllm_wsl.sh) and returns correct gencode flags.
#   The traceback confirms get_nvcc_flags_list() returns ONLY arch/gencode flags;
#   all other compile flags arrive via common_cflags / extra_cuda_cflags in
#   build_cuda_cflags(), so returning just the gencode entry is correct.

import re
import sys
import pathlib
import inspect

try:
    import flashinfer.jit.core as jit_core_mod
except ImportError:
    print('flashinfer not installed, skipping patches')
    sys.exit(0)

# ------------------------------------------------------------------
# Patch 1: check_cuda_arch in flashinfer/jit/core.py
# ------------------------------------------------------------------
p1 = pathlib.Path(inspect.getsourcefile(jit_core_mod))
orig1 = p1.read_text()

# Idempotency: the replacement body below contains this exact line, so its
# presence means patch 1 was already applied on a previous run.
_P1_MARKER = 'sm = major * 10 + minor'

repl1 = (
    'def check_cuda_arch():\n'
    '    import torch\n'
    '    if not torch.cuda.is_available():\n'
    '        return\n'
    '    major, minor = torch.cuda.get_device_capability()\n'
    '    sm = major * 10 + minor\n'
    '    if sm < 75:\n'
    '        raise RuntimeError(\n'
    '            "FlashInfer requires sm75+, detected sm{}" .format(sm))\n'
    '\n'
)

if _P1_MARKER in orig1:
    print('Patch 1 already present in:', p1)
else:
    p1_patched = re.sub(
        r'def check_cuda_arch\(\):.*?(?=\ndef |\Z)',
        repl1,
        orig1,
        flags=re.DOTALL,
    )
    if p1_patched == orig1:
        print('WARNING: patch 1 (check_cuda_arch) pattern not found in', p1)
        sys.exit(1)
    p1.write_text(p1_patched)
    print('Patch 1 applied: check_cuda_arch in', p1)


# ------------------------------------------------------------------
# Patch 2: CompilationContext.get_nvcc_flags_list in compilation_context.py
#
# Appends a monkey-patch block to the module.  The block runs at import
# time, finds whichever class owns get_nvcc_flags_list, and wraps it.
# When FLASHINFER_CUDA_ARCHS="12.0" is in the environment the wrapper
# returns ['-gencode', 'arch=compute_120,code=sm_120'] immediately,
# bypassing the broken lookup table.  Otherwise it delegates to the
# original (so non-Blackwell systems are unaffected).
# ------------------------------------------------------------------
import flashinfer.compilation_context as cc_mod
p2 = pathlib.Path(inspect.getsourcefile(cc_mod))
orig2 = p2.read_text()

marker = '# --- Blackwell sm_120 patch ---'
if marker in orig2:
    print('Patch 2 already present in:', p2)
else:
    monkey = (
        '\n\n'
        + marker + '\n'
        'import os as _p_os, sys as _p_sys\n'
        '_p_mod = _p_sys.modules[__name__]\n'
        'for _p_name in list(vars(_p_mod)):\n'
        '    _p_cls = vars(_p_mod)[_p_name]\n'
        '    if isinstance(_p_cls, type) and hasattr(_p_cls, "get_nvcc_flags_list"):\n'
        '        _p_orig = _p_cls.get_nvcc_flags_list\n'
        '        def _p_patched(self, _orig=_p_orig):\n'
        '            env = _p_os.environ.get("FLASHINFER_CUDA_ARCHS", "")\n'
        '            if env:\n'
        '                flags = []\n'
        '                for a in env.replace(";", ",").split(","):\n'
        '                    a = a.strip().replace(".", "")\n'
        '                    if a:\n'
        '                        flags.extend(\n'
        '                            ["-gencode",\n'
        '                             "arch=compute_{0},code=sm_{0}".format(a)])\n'
        '                if flags:\n'
        '                    return flags\n'
        '            return _orig(self)\n'
        '        _p_cls.get_nvcc_flags_list = _p_patched\n'
        '        print("Patch 2 active: wrapped get_nvcc_flags_list on", _p_name)\n'
        '        break\n'
        '# --- end Blackwell sm_120 patch ---\n'
    )
    p2.write_text(orig2 + monkey)
    print('Patch 2 applied: monkey-patched get_nvcc_flags_list in', p2)
