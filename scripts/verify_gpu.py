# scripts/verify_gpu.py
# Run this before anything else to confirm the GPU environment is correct.
# Usage: python3 scripts/verify_gpu.py

import sys
import subprocess


def check(label, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    line = f'  [{status}] {label}'
    if detail:
        line += f' -- {detail}'
    print(line)
    return condition


def main():
    print('\nBoard Intelligence System -- GPU Environment Verification')
    print('=' * 60)

    all_pass = True

    # Check PyTorch
    print('\n[1] PyTorch + CUDA')
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        all_pass &= check('CUDA available', cuda_available)

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)

            all_pass &= check(
                'Device name',
                'RTX 5090' in device_name or '5090' in device_name,
                device_name
            )
            all_pass &= check(
                'Compute capability sm_120',
                cc == (12, 0),
                f'sm_{cc[0]}{cc[1]}'
            )
            all_pass &= check(
                'VRAM >= 20 GB',
                vram >= 20000,
                f'{vram} MiB'
            )
            all_pass &= check(
                'PyTorch version',
                tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (2, 9),
                torch.__version__
            )

            # Verify CUDA 12.8 build
            cuda_version = torch.version.cuda
            all_pass &= check(
                'PyTorch built with CUDA 12.8',
                cuda_version is not None and cuda_version.startswith('12.8'),
                f'cuda {cuda_version}'
            )

    except ImportError:
        all_pass &= check('PyTorch importable', False, 'not installed')

    # Check CUDA toolkit
    print('\n[2] CUDA Toolkit')
    import platform
    on_windows = platform.system() == 'Windows'
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True, text=True
        )
        nvcc_out = result.stdout + result.stderr
        has_nvcc = result.returncode == 0
        is_128 = 'release 12.8' in nvcc_out or 'V12.8' in nvcc_out
        nvcc_line = nvcc_out.split('\n')[3].strip() if has_nvcc else 'not found'
        check('nvcc available', has_nvcc, nvcc_line)
        if has_nvcc:
            if on_windows and not is_128:
                # Windows system nvcc is not used by the pipeline. vLLM runs
                # in WSL with a user-local CUDA 12.8 toolkit at ~/cuda-12.8
                # installed by scripts/setup_cuda128_wsl.sh.
                print(f'  [INFO] nvcc is {nvcc_line} -- OK on Windows; WSL uses ~/cuda-12.8 for vLLM')
            else:
                all_pass &= check('nvcc is CUDA 12.8', is_128, 'CRITICAL: must be 12.8 not 13.x')
    except FileNotFoundError:
        print('  [INFO] nvcc not found in PATH -- expected if running inside container')

    # Check nvidia-smi
    print('\n[3] nvidia-smi')
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f'  [INFO] {gpu_info}')
            check('nvidia-smi accessible', True)
        else:
            check('nvidia-smi accessible', False, result.stderr)
    except FileNotFoundError:
        print('  [INFO] nvidia-smi not in PATH -- expected in WSL2')

    # Check WSL2 version hint
    print('\n[4] WSL2 / System')
    try:
        with open('/proc/version', 'r') as f:
            proc_version = f.read().strip()
        is_wsl = 'microsoft' in proc_version.lower() or 'wsl' in proc_version.lower()
        check('Running in WSL2', is_wsl, proc_version[:80])
    except Exception:
        print('  [INFO] /proc/version not readable')

    # Check key libraries
    print('\n[5] Key Libraries')
    libs = [
        ('docling', 'docling'),
        ('FlagEmbedding', 'FlagEmbedding'),
        ('sentence_transformers', 'sentence_transformers'),
        ('qdrant_client', 'qdrant_client'),
        ('rank_bm25', 'rank_bm25'),
        ('fastapi', 'fastapi'),
        ('rapidocr_onnxruntime', 'rapidocr_onnxruntime'),
    ]
    for display, module in libs:
        try:
            imported = __import__(module)
            ver = getattr(imported, '__version__', 'unknown')
            check(f'{display}', True, ver)
        except ImportError:
            check(f'{display}', False, 'not installed')

    # Summary
    print('\n' + '=' * 60)
    if all_pass:
        print('RESULT: ALL CHECKS PASSED -- environment is ready')
        print('You can now run: scripts\\start_all.ps1')
    else:
        print('RESULT: SOME CHECKS FAILED -- resolve issues above before proceeding')
        print('')
        print('Common fixes:')
        print('  WSL2 too old: wsl --update  (in PowerShell)')
        print('  CUDA 12.8 missing in WSL: bash scripts/setup_cuda128_wsl.sh')
        print('  Libraries missing: pip install -r requirements.txt')
    print('')

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
