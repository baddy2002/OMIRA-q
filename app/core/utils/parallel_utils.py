# app/core/utils/parallel_utils.py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import os
import threading
import atexit

class ExecutorSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_workers=8):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ExecutorSingleton, cls).__new__(cls)
                    cls._instance.executor = ThreadPoolExecutor(max_workers=max_workers)
        return cls._instance

    @classmethod
    def get_executor(cls, max_workers=8):
        if cls._instance is None:
            cls._instance = cls(max_workers=max_workers)
        return cls._instance._instance.executor if hasattr(cls._instance, '_instance') else cls._instance.executor


class GPUProcessPoolSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_workers=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Su Kaggle ci sono 4 vCPU: 4 worker è il compromesso ideale tra CPU e VRAM
                    workers = max_workers or min(4, os.cpu_count() or 2)
                    cls._instance = super().__new__(cls)
                    ctx = mp.get_context("spawn")
                    cls._instance.executor = ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=ctx
                    )
                    atexit.register(cls._instance.executor.shutdown, wait=False)
        return cls._instance

    @classmethod
    def get_executor(cls, max_workers=None):
        return cls(max_workers=max_workers).executor