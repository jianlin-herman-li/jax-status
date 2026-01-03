import ctypes
import os
import platform

if platform.system().lower() == "linux":
    driver_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    ]
    for base in driver_paths:
        path = os.path.join(base, "libcuda.so.1")
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                os.environ["JAX_STATUS_LIBCUDA_PRELOADED"] = "1"
            except Exception as exc:
                os.environ["JAX_STATUS_LIBCUDA_PRELOAD_ERROR"] = str(exc)
                pass
            try:
                import subprocess
                subprocess.getoutput("nvidia-smi")
            except Exception:
                pass
            break
