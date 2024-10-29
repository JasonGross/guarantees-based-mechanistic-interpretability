#!/usr/bin/env python3
# %%
import os
import subprocess
import sys
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


def check_source_in_git_dir(git_dir: str | Path, source: str = "max_of_4.py") -> bool:
    """Checks if source is in the GIT_DIR path."""
    return source in str(git_dir)


def get_objects_path(git_dir: str | Path) -> Path:
    """Returns the path to the alternates file, or None if it does not exist."""
    return Path(git_dir) / "objects"


def get_alternates_file(git_dir: str | Path) -> Path:
    """Returns the path to the alternates file, or None if it does not exist."""
    return get_objects_path(git_dir) / "info" / "alternates"


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


def process_submodules(
    sources: dict[str, str] = {
        ".py": "max_of_4.py",
        "_all_models": "max_of_4_all_models",
    }
) -> None:
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
    any_git_dirs = {submodule: get_git_dir(submodule) for submodule in submodules}
    empty_git_dirs = {
        submodule for submodule, git_dir in any_git_dirs.items() if git_dir is None
    }
    git_dirs = {
        submodule: git_dir
        for submodule, git_dir in any_git_dirs.items()
        if git_dir is not None
    }
    if empty_git_dirs:
        print(
            f"Could not find .git directories for submodules: {tuple(sorted(empty_git_dirs))}"
        )
    source_git_dirs = {
        key: {
            submodule: git_dir
            for submodule, git_dir in git_dirs.items()
            if check_source_in_git_dir(git_dir, value)
        }
        for key, value in sources.items()
    }
    other_git_dirs = {
        key: {
            submodule: git_dir
            for submodule, git_dir in git_dirs.items()
            if submodule not in source_git_dirs[key] and key in submodule
        }
        for key in sources
    }
    unknown_git_dirs = {
        submodule: git_dir
        for submodule, git_dir in git_dirs.items()
        if all(
            submodule not in source_git_dirs[key]
            and submodule not in other_git_dirs[key]
            for key in sources
        )
    }
    if unknown_git_dirs:
        print(f"Unknown submodules: {unknown_git_dirs}")
        sys.exit(1)
    source_git_dir = {}
    for key, cur_source_git_dirs in source_git_dirs.items():
        if len(cur_source_git_dirs) == 1:
            _source_submodule, source_git_dir[key] = list(cur_source_git_dirs.items())[
                0
            ]
        elif not cur_source_git_dirs:
            print(f"Could not find {key} in any GIT_DIR path ({any_git_dirs})")
            sys.exit(1)
        else:
            print(
                f"Found {len(cur_source_git_dirs)} instances of {key} in GIT_DIR paths ({cur_source_git_dirs})"
            )
            sys.exit(1)

    relative_targets = {
        key: get_objects_path(git_dir) for key, git_dir in source_git_dir.items()
    }

    for key, cur_other_git_dirs in other_git_dirs.items():
        for submodule, git_dir in cur_other_git_dirs.items():
            print(f"Checking submodule: {submodule}")

            print(f"{key} not found in GIT_DIR: {git_dir}")

            # Check for the alternates file
            alternates_file = get_alternates_file(git_dir)
            print(f"Alternates file found at: {alternates_file}")

            # Calculate the relative path to max_of_4.py/objects using pathlib
            print(
                f"Finding get_relative_path({alternates_file.parent.parent}, {relative_targets[key]})"
            )
            relative_path = get_relative_path(
                alternates_file.parent.parent, relative_targets[key]
            )
            print(f"Relative path to alternates: {relative_path}")
            alternates_file.write_text(str(relative_path))


if __name__ == "__main__":
    process_submodules()
