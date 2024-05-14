import threading
import gc
import time


class PeriodicGarbageCollector:
    def __init__(self, sleep_seconds: float, collect_on_exit: bool = True):
        self.sleep_seconds = sleep_seconds
        self.stop_gc = False
        self.collect_on_exit = collect_on_exit
        self.gc_thread = threading.Thread(target=self._garbage_collector)

    def _garbage_collector(self):
        while not self.stop_gc:
            gc.collect()
            time.sleep(self.sleep_seconds)

    def __enter__(self):
        self.stop_gc = False
        self.gc_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_gc = True
        self.gc_thread.join()
        if self.collect_on_exit:
            gc.collect()
