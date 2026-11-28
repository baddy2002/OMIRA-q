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


def _init_gpu_worker(gpu_queue):
    """Eseguito una sola volta all'avvio di ciascun processo figlio."""
    # Evita che ogni worker crei 4 thread OpenMP bloccando le vCPU di Kaggle
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if gpu_queue is not None:
        gpu_id = gpu_queue.get()
        # Isola il processo sulla singola scheda assegnata (0 oppure 1 su Kaggle T4 x2)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.set_num_threads(1)

class GPUProcessPoolSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_workers=4):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    import torch
                    cls._instance = super().__new__(cls)
                    ctx = mp.get_context("spawn")

                    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    if num_gpus > 0:
                        # Distribuisce i worker equamente su tutte le GPU disponibili (es. 2 su T4_0, 2 su T4_1)
                        gpu_queue = ctx.Queue()
                        for w in range(max_workers):
                            gpu_queue.put(w % num_gpus)
                        cls._instance.executor = ProcessPoolExecutor(
                            max_workers=max_workers,
                            mp_context=ctx,
                            initializer=_init_gpu_worker,
                            initargs=(gpu_queue,),
                        )
                    else:
                        # Fallback pulito per Mac (MPS/CPU)
                        cls._instance.executor = ProcessPoolExecutor(
                            max_workers=max_workers,
                            mp_context=ctx,
                            initializer=_init_gpu_worker,
                            initargs=(None,),
                        )
                    atexit.register(cls._instance.executor.shutdown, wait=False)
        return cls._instance

    @classmethod
    def get_executor(cls, max_workers=4):
        return cls(max_workers=max_workers).executor