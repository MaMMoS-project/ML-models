from contextlib import redirect_stdout

def log_output(file_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with open(file_path, "a") as f:
                with redirect_stdout(f):
                    return func(*args, **kwargs)
        return wrapper
    return decorator
