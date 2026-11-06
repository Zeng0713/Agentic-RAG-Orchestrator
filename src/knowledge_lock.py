"""Writer-preferring, process-local read/write lock for the local knowledge store."""

from contextlib import contextmanager
import threading


class KnowledgeLock:
    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0
        self._writing = False
        self._waiting_writers = 0

    @contextmanager
    def read(self):
        with self._condition:
            self._condition.wait_for(lambda: not self._writing and not self._waiting_writers)
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                self._condition.notify_all()

    @contextmanager
    def write(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                self._condition.wait_for(lambda: not self._writing and not self._readers)
                self._writing = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writing = False
                self._condition.notify_all()
