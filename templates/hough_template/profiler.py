import time
import tracemalloc
from enum import Enum


class TimeUnit(Enum):
    s = 1.0
    ms = 1e-3
    us = 1e-6
    ns = 1e-9

    def __init__(self, mult):
        self.mult = mult


class Timer:
    def __init__(self, unit=TimeUnit.ms):
        self.start_time = time.time()
        self.unit = unit

    def stop(self):
        elapsed_time = (time.time() - self.start_time)  # in sec
        return elapsed_time / self.unit.mult

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = self.stop()
        print(f'Elapsed time: {elapsed_time:0.2f} {self.unit.name}')


class MemoryUnit(Enum):
    B = 1
    KiB = 2 ** 10
    MiB = 2 ** 20
    GiB = 2 ** 30

    def __init__(self, mult):
        self.mult = mult


class MemoryAnalyzer:
    def __init__(self, unit=MemoryUnit.KiB):
        tracemalloc.start()
        self.start_snapshot = tracemalloc.take_snapshot()
        self.unit = unit

    def stop(self):
        final_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = final_snapshot.compare_to(self.start_snapshot, 'lineno')
        allocated_memory = sum(stat.size for stat in stats)
        return allocated_memory / self.unit.mult

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        allocated_memory = self.stop()
        print(f'Allocated memory: {allocated_memory:0.2f} {self.unit.name}')


class Profiler:
    def __init__(self, time_unit=TimeUnit.ms, memory_unit=MemoryUnit.KiB):
        self.timer = Timer(time_unit)
        self.memory_analyzer = MemoryAnalyzer(memory_unit)

    def stop(self):
        elapsed_time = self.timer.stop()
        allocated_memory = self.memory_analyzer.stop()
        return elapsed_time, allocated_memory

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.__exit__(exc_type, exc_val, exc_tb)
        self.memory_analyzer.__exit__(exc_type, exc_val, exc_tb)
