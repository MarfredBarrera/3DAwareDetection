"""Disable transparent huge pages in every Python process in the container.

Python imports this at startup because /workspace is on PYTHONPATH in every
env. On the shared host no NUMA node has free 2 MB blocks, so a huge-page
fault stalls in direct compaction (seconds per MB). The stall counts as
memory pressure, and systemd-oomd then SIGKILLs a whole container. numpy
madvises huge pages for large arrays by default, so any process can trigger it.
PR_SET_THP_DISABLE is inherited by child processes, so it also covers them.
"""
import ctypes
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
PR_SET_THP_DISABLE = 41
if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
    print(f"sitecustomize: PR_SET_THP_DISABLE failed (errno {ctypes.get_errno()})", flush=True)
