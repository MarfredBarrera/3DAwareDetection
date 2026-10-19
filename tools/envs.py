"""Run workers in the two model environments on one configured GPU.

SAM3 (``core``) and SAM3D/PyTorch3D (``sam3d``) need incompatible PyTorch
builds, so every GPU stage is a subprocess launched through ``sam3d-env``.
"""
import os
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]


def no_huge_pages():
    """Disable transparent huge pages for this process and its children.

    THP faults on the shared host stall in direct compaction; the stall counts
    as memory pressure and systemd-oomd then kills the whole container.
    """
    import ctypes
    PR_SET_THP_DISABLE = 41
    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        print(f"warning: PR_SET_THP_DISABLE failed (errno {ctypes.get_errno()})", flush=True)


def use_gpu(gpu_id):
    """Restrict this process to one host GPU; call before importing torch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def run_in_env(env, module, args, *, gpu_id, log=None):
    """Run ``python -m module args`` in ``env``, streaming output to stdout and ``log``."""
    command = ["sam3d-env", env, "python", "-m", module, *map(str, args)]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
                   # Huge-page advice on repeated mask allocations stalled the shared
                   # host's cgroup until systemd-oomd killed the container.
                   "NUMPY_MADVISE_HUGEPAGE": os.environ.get("NUMPY_MADVISE_HUGEPAGE", "0")}
    print(f"$ {' '.join(command)}", flush=True)
    stream = open(log, "a", buffering=1) if log else None
    try:
        process = subprocess.Popen(command, cwd=WORKSPACE, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            print(line, end="", flush=True)
            if stream:
                stream.write(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if stream:
            stream.close()
