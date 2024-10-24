#!/usr/bin/env python3
# %%
import os
import subprocess
from pathlib import Path


def get_git_dir(submodule_path: str | Path) -> Path | None:
    """Gets the actual .git directory for a submodule."""
    git_dir_file = Path(submodule_path) / ".git"
    if git_dir_file.is_file():
        with git_dir_file.open("r") as f:
            git_dir_relative = f.read().strip().split(": ")[1]
        git_dir = Path(submodule_path) / git_dir_relative
        return git_dir.resolve()
    return None


def check_max_of_4_in_git_dir(git_dir: str | Path) -> bool:
    """Checks if 'max_of_4.py' is in the GIT_DIR path."""
    return "max_of_4.py" in str(git_dir)


def get_alternates_file(git_dir: str | Path) -> Path | None:
    """Returns the path to the alternates file, or None if it does not exist."""
    return Path(git_dir) / "objects" / "info" / "alternates"


def get_relative_path(submodule_path: str | Path, target_path: str | Path) -> Path:
    """Calculates the relative path from the submodule's root directory to the target path."""
    submodule_path = Path(submodule_path).resolve()
    target_path = Path(target_path).resolve()
    return Path(os.path.relpath(target_path, submodule_path))


def get_repo_root() -> Path:
    """Returns the root directory of the git repository."""
    # Run the git command to get the top-level directory
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Return the path as a Path object
    return Path(result.stdout.strip())


def process_submodules():
    """Processes each submodule and checks for max_of_4.py, alternates, and relative paths."""
    # Use git submodule foreach to get the list of submodules
    os.chdir(get_repo_root())
    result = subprocess.run(
        ["git", "submodule", "foreach", "--quiet", "echo $path"],
        capture_output=True,
        text=True,
        check=True,
    )
    submodules = result.stdout.strip().splitlines()

    for submodule in submodules:
        print(f"Checking submodule: {submodule}")

        # Get the actual git directory for the submodule
        git_dir = get_git_dir(submodule)
        if git_dir is None:
            print(f"Could not find .git directory for submodule: {submodule}")
            continue

        # Check if 'max_of_4.py' is in the GIT_DIR path
        if not check_max_of_4_in_git_dir(git_dir):
            print(f"max_of_4.py not found in GIT_DIR: {git_dir}")

            # Check for the alternates file
            alternates_file = get_alternates_file(git_dir)
            print(f"Alternates file found at: {alternates_file}")

            # Calculate the relative path to max_of_4.py/objects using pathlib
            relative_target = ".git/modules/notebooks_jason/.cache/max_of_4.py/objects"
            print(f"Finding get_relative_path({alternates_file.parent.parent}, {relative_target})")
            relative_path = get_relative_path(alternates_file.parent.parent, relative_target)
            print(f"Relative path to alternates: {relative_path}")
            alternates_file.write_text(str(relative_path))
        else:
            print("max_of_4.py is already in the GIT_DIR path")


if __name__ == "__main__":
    process_submodules()
