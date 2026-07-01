import os
import sys


class _Tee:
    """Forward writes to several streams safely."""

    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")

        alive_streams = []

        for s in self.streams:
            try:
                if s is None:
                    continue
                if hasattr(s, "closed") and s.closed:
                    continue

                s.write(data)
                s.flush()
                alive_streams.append(s)

            except (ValueError, OSError, AttributeError):
                # stream is dead → drop it
                continue

        self.streams = alive_streams
        return len(data)

    def flush(self):
        alive_streams = []

        for s in self.streams:
            try:
                if s is None:
                    continue
                if hasattr(s, "closed") and s.closed:
                    continue

                s.flush()
                alive_streams.append(s)

            except (ValueError, OSError, AttributeError):
                continue

        self.streams = alive_streams
        

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
