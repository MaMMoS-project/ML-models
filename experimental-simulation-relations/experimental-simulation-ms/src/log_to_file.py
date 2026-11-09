import os
import sys
from contextlib import redirect_stdout


class Tee:
    """A stdout-like stream that writes to several underlying streams at once,
    flushing after every write so the log file is updated continuously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def log_output(file_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # make sure the target directory exists
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            # Truncate per run ("w") so each run starts a fresh log instead of
            # appending onto previous runs; line-buffered (buffering=1) so each
            # line is flushed to disk continuously. Each log file has a single
            # decorated entry point, so per-run truncation is safe.
            with open(file_path, "w", buffering=1) as f:
                tee = Tee(sys.stdout, f)
                with redirect_stdout(tee):
                    result = func(*args, **kwargs)

            return result
        return wrapper
    return decorator
