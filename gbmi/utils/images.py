import io
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import plotly.graph_objects as go
from PIL import Image, ImageChops
from tqdm.auto import tqdm


def trim(im: Image.Image, border_color=None) -> Image.Image:
    if border_color is None:
        border_color = im.getpixel((0, 0))
    bg = Image.new(im.mode, im.size, border_color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox(alpha_only=False)
    if bbox:
        return im.crop(bbox)
    raise ValueError("cannot trim; bounding box was empty")


def trim_plotly_figure(
    fig: go.Figure, border_color=None, pad: int = 0, padh: int = 0, padw: int = 0
) -> go.Figure:
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    image = Image.open(io.BytesIO(fig.to_image("png")))
    cropped_image = trim(image, border_color=border_color)
    fig.update_layout(
        width=cropped_image.width + pad + padw, height=cropped_image.height + pad + padh
    )
    return fig


def remove_bak(*files: Union[str, Path], save_bak: bool = True, ext: str = ".bak"):
    file_paths = [Path(file) for file in files]
    bak_files = [file.with_suffix(file.suffix + ext) for file in file_paths]
    extant_bak_files = [bak_file for bak_file in bak_files if bak_file.exists()]
    if save_bak:
        if extant_bak_files:
            remove_bak(*extant_bak_files, save_bak=save_bak, ext=ext)
        for bak_file in extant_bak_files:
            bak_file.rename(bak_file.with_suffix(bak_file.suffix + ext))
    else:
        for bak_file in extant_bak_files:
            bak_file.unlink()


def forward_output(
    stream: io.BufferedReader,
    write_func: Callable[[str], None],
    trim_func: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    buffer = ""
    for char in iter(lambda: stream.read(1), b""):
        decoded_char = char.decode()
        if trim_func is None:
            write_func(decoded_char, end="")
        else:
            buffer += decoded_char
            if "\n" in buffer:
                lines = buffer.splitlines(keepends=True)
                for line in lines[:-1]:
                    trimmed_line = trim_func(line)
                    if trimmed_line is not None:
                        write_func(trimmed_line, end="")
                buffer = lines[-1]


def batch_run(
    args: Iterable[str],
    *images: str,
    batchsize: int = 64,
    post_args: list[str] = [],
    check: bool = True,
    trim_printout: Optional[Callable[[str], Optional[str]]] = None,
    stdout_write: Optional[Callable] = None,
    stderr_write: Optional[Callable] = None,
    **kwargs,
):
    if len(images) > batchsize:
        return [
            batch_run(
                args,
                *images[i : i + batchsize],
                batchsize=batchsize,
                post_args=post_args,
                check=check,
                **kwargs,
            )
            for i in range(0, len(images), batchsize)
        ]

    process = subprocess.Popen(
        [*args, *images, *post_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )

    if stdout_write is None:
        stdout_write = partial(print, file=sys.stdout)
    if stderr_write is None:
        stderr_write = partial(print, file=sys.stderr)

    stdout_thread = threading.Thread(
        target=forward_output, args=(process.stdout, stdout_write, trim_printout)
    )
    stderr_thread = threading.Thread(
        target=forward_output, args=(process.stderr, stderr_write, trim_printout)
    )

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    process.wait()

    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)

    return process


def trim_ect(output: str) -> Optional[str]:
    if output.strip() == "Processed 1 file":
        return None
    return output


def ect(
    *images: Union[str, Path],
    level: Optional[int] = None,
    exhaustive: bool = False,
    strip: bool = True,
    strict: bool = True,
    extra_args: Sequence[str] = (),
    trim_printout: bool = False,
    stdout_write: Optional[Callable] = None,
    stderr_write: Optional[Callable] = None,
):
    if not images:
        return
    if level is None and exhaustive:
        level = 9
    extra_args = list(extra_args)
    if level is not None:
        extra_args.append(f"-{level}")
    if strip:
        extra_args.append("-strip")
    if strict:
        extra_args.append("--strict")
    return batch_run(
        ["ect", *extra_args],
        *images,
        check=True,
        trim_printout=trim_ect if trim_printout else None,
        stdout_write=stdout_write,
        stderr_write=stderr_write,
    )


def trim_optipng(output: str) -> Optional[str]:
    if output.strip() == "Trying:":
        return None
    pattern = r"zc = \d+ zm = \d+ zs = \d+ f = \d+"
    regex_pattern = pattern.replace(" ", r"\s*")
    if re.match(regex_pattern, output.strip()):
        return None
    return output


def optipng(
    *images: Union[str, Path],
    level: int = 5,
    exhaustive: bool = False,
    save_bak: bool = True,
    fix: bool = True,
    trim_printout: bool = False,
    stdout_write: Optional[Callable] = None,
    stderr_write: Optional[Callable] = None,
):
    if not images:
        return
    if level == 5 and exhaustive:
        level = 7
    extra_args = [] if not exhaustive else ["-zm1-9"]
    extra_args += ["-fix"] if fix else []
    remove_bak(*images, save_bak=save_bak)
    return batch_run(
        ["optipng", f"-o{level}", *extra_args],
        *images,
        check=True,
        trim_printout=trim_optipng if trim_printout else None,
        stdout_write=stdout_write,
        stderr_write=stderr_write,
    )


def trim_pngcrush(output: str) -> Optional[str]:
    return output


def pngcrush(
    *images: Union[str, Path],
    brute: bool = True,
    tmpdir: Optional[Union[str, Path]] = None,
    cleanup: Optional[bool] = None,
    trim_printout: bool = False,
    stdout_write: Optional[Callable] = None,
    stderr_write: Optional[Callable] = None,
):
    if not images:
        return
    if len(images) > 1 and len(images) != len(
        set(Path(image).name for image in images)
    ):
        return [
            pngcrush(image, brute=brute, tmpdir=tmpdir, cleanup=cleanup)
            for image in images
        ]

    if not tmpdir:
        tmpdir = tempfile.TemporaryDirectory(dir=Path(images[0]).parent)
        tmpdir_path = Path(tmpdir.name)
        if cleanup is None or cleanup is True:
            cleanup = tmpdir.cleanup
    else:
        tmpdir_path = Path(tmpdir)
        if cleanup is True or (cleanup is None and not tmpdir_path.exists()):
            cleanup = tmpdir_path.rmdir()
        tmpdir_path.mkdir(parents=True, exist_ok=True)

    args = ["pngcrush"]
    args += ["-brute"] if brute else []
    args += ["-d", str(tmpdir_path)]

    # Run pngcrush with the specified arguments
    batch_run(
        args,
        *map(str, images),
        check=True,
        trim_printout=trim_pngcrush if trim_printout else None,
        stdout_write=stdout_write,
        stderr_write=stderr_write,
    )

    # Replace original images with crushed images if they are smaller
    for old_f in map(Path, images):
        new_f = tmpdir_path / old_f.name
        if new_f.exists() and new_f.stat().st_size < old_f.stat().st_size:
            new_f.rename(old_f)
        elif new_f.exists():
            new_f.unlink()

    if cleanup:
        cleanup()


def optimize(
    *images: Union[str, Path],
    exhaustive: bool = True,
    tmpdir: Optional[Union[str, Path]] = None,
    cleanup: Optional[bool] = None,
    trim_printout: bool = False,
    tqdm_position: Optional[int] = None,
    tqdm_leave: Optional[bool] = None,
    stdout_write: Optional[Callable] = None,
    stderr_write: Optional[Callable] = None,
):
    cur_images = images
    cur_sizes = [Path(image).stat().st_size for image in cur_images]
    while cur_images:
        if shutil.which("ect"):
            for img in tqdm(
                cur_images, desc="ect", position=tqdm_position, leave=tqdm_leave
            ):
                ect(
                    img,
                    exhaustive=exhaustive,
                    trim_printout=trim_printout,
                    stdout_write=partial(tqdm.write, file=sys.stdout),
                    stderr_write=partial(tqdm.write, file=sys.stderr),
                )
        optipng(
            *cur_images,
            exhaustive=exhaustive,
            trim_printout=trim_printout,
            stdout_write=stdout_write,
            stderr_write=stderr_write,
        )
        pngcrush(
            *cur_images,
            brute=exhaustive,
            tmpdir=tmpdir,
            cleanup=cleanup,
            trim_printout=trim_printout,
            stdout_write=stdout_write,
            stderr_write=stderr_write,
        )
        new_sizes = [Path(image).stat().st_size for image in cur_images]
        cur_images = [
            image
            for image, old_size, new_size in zip(cur_images, cur_sizes, new_sizes)
            if new_size < old_size
        ]
        cur_sizes = [Path(image).stat().st_size for image in cur_images]
