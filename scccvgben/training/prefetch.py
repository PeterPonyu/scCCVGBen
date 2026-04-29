"""prefetch.py — Overlap CPU preprocessing with GPU training.

Provides ``DatasetPrefetcher``: submits preprocessing futures for all paths at
construction time so GPU training can call ``get(path)`` and block only on the
specific path it needs, rather than waiting for preprocessing serially.

Typical usage (future wiring into run_encoder_sweep.py):

    from scccvgben.training.prefetch import DatasetPrefetcher

    prefetcher = DatasetPrefetcher(dataset_paths, preprocess_fn, max_workers=4)
    for path in dataset_paths:
        adata = prefetcher.get(path)   # returns immediately if already done
        train(adata)

Only stdlib is used (concurrent.futures, pathlib).
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


class DatasetPrefetcher:
    """Submit preprocessing futures eagerly; retrieve results on demand.

    Parameters
    ----------
    dataset_paths : list[Path]
        Ordered list of dataset file paths to preprocess.
    preprocess_fn : callable
        Function ``(Path) -> Any`` that loads and preprocesses one dataset.
        Must be thread-safe (e.g. pure I/O + CPU numpy; no CUDA).
    max_workers : int
        Thread-pool size (default 4). I/O-bound work scales well with threads;
        CPU-bound work should use a smaller value or a ProcessPool instead.
    """

    def __init__(
        self,
        dataset_paths: list[Path],
        preprocess_fn: Callable[[Path], Any],
        max_workers: int = 4,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[Path, Future] = {}
        for path in dataset_paths:
            p = Path(path)
            self._futures[p] = self._executor.submit(preprocess_fn, p)

    def get(self, path: Path) -> Any:
        """Block until preprocessing for *path* is complete; return the result.

        Raises the original exception if preprocessing failed for this path.
        """
        p = Path(path)
        future = self._futures[p]
        return future.result()  # re-raises any exception from the worker thread

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool (optional; also called on GC)."""
        self._executor.shutdown(wait=wait)

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


if __name__ == "__main__":
    import time

    results: list[str] = []

    def _fake_preprocess(p: Path) -> str:
        time.sleep(0.05)  # simulate I/O
        return f"processed:{p}"

    paths = [Path("a"), Path("b")]
    pf = DatasetPrefetcher(paths, _fake_preprocess, max_workers=2)

    for path in paths:
        result = pf.get(path)
        assert result == f"processed:{path}", f"unexpected: {result!r}"
        results.append(result)

    pf.shutdown()
    assert len(results) == 2
    print("prefetch self-test passed:", results)
