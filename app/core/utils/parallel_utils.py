# app/core/utils/parallel_utils.py
from concurrent.futures import ThreadPoolExecutor
import threading

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