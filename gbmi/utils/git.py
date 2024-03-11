import subprocess


def rev_parse(rev: str, *, short: bool = False) -> str:
    cmd = ["git", "rev-parse", "--short", rev] if short else ["git", "rev-parse", rev]
    return subprocess.check_output(cmd).decode("utf-8").strip()


def get_head_sha(*, short: bool = False) -> str:
    return rev_parse("HEAD", short=short)


def get_diff() -> str:
    return subprocess.check_output(["git", "--no-pager", "diff"]).decode("utf-8")
