import subprocess

def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"