import io
from contextlib import redirect_stdout

def log_output(file_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                result = func(*args, **kwargs)

            with open(file_path, "a") as f:
                f.write(buffer.getvalue())

            return result
        return wrapper
    return decorator
