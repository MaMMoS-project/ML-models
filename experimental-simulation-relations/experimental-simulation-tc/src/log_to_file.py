import os
import sys


class _Tee:
    """Forward writes to several streams at once, flushing after each write.

    Used so stdout is shown live in the terminal / SLURM .out file AND written
    to the log file at the same time, line by line, rather than buffered until
    the decorated function returns.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def log_output(file_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            original_stdout = sys.stdout
            # "w" truncates the log at the start of every run (one run per file);
            # buffering=1 makes the file line-buffered for continuous output.
            with open(file_path, "w", buffering=1) as f:
                sys.stdout = _Tee(original_stdout, f)
                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = original_stdout

            return result
        return wrapper
    return decorator
