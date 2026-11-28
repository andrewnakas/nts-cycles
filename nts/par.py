"""Frame-parallel helpers (2026-11-24 speed pass).

Most per-frame work in the NR-1 route (the tone step's two passes, the carry's compose) is independent from frame to
frame once the shot's constants are known. pmap() runs it in worker processes and returns results in input order, so
the frames written are the same files with the same bytes as a sequential run; only the wall-clock time changes.

  workers(n)   n > 0: n processes; n == 0: one per CPU core; None: env NTS_WORKERS, else 1 (sequential, as before).
  pmap(fn, items, n, init=None, initargs=())   ordered map; fn and init must be module-level functions.

Linux forks the workers (shared memory, nothing to pickle up front); macOS and Windows spawn them, so a task's
arguments and the initializer's arguments are pickled once per task / per worker.
"""
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys


def workers(n=None):
    if n is None:
        n = os.environ.get("NTS_WORKERS", "1")
    n = int(n)
    return max(1, os.cpu_count() or 1) if n <= 0 else max(1, n)


def _ctx():
    if sys.platform.startswith("linux"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def pmap(fn, items, n=1, init=None, initargs=()):
    """Ordered map of fn over items, in n worker processes (n <= 1 or a single item: in this process)."""
    items = list(items)
    n = min(workers(n), len(items))
    if n <= 1:
        if init is not None:
            init(*initargs)
        return [fn(it) for it in items]
    with cf.ProcessPoolExecutor(n, mp_context=_ctx(), initializer=init, initargs=initargs) as ex:
        return list(ex.map(fn, items, chunksize=1))
