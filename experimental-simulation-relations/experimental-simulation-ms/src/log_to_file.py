import os
import sys


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")

        for s in self.streams:
            if not s.closed:
                s.write(data)
                s.flush()

        return len(data)

    def flush(self):
        for s in self.streams:
            if not s.closed:
                s.flush()


def log_output(file_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            original_stdout = sys.stdout

            with open(file_path, "w", buffering=1) as f:
                sys.stdout = _Tee(original_stdout, f)

                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = original_stdout

            return result

        return wrapper
    return decorator
