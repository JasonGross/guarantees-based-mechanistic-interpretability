from typing import Union, Optional, Sequence
import shutil
import tempfile
import subprocess
from pathlib import Path
from PIL import Image, ImageChops
import io
import plotly.graph_objects as go
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


def batch_run(args, *images, batchsize=64, post_args=[], **kwargs):
    if len(images) > batchsize:
        return [
            batch_run(
                args,
                *images[i : i + batchsize],
                batchsize=batchsize,
                post_args=post_args,
                **kwargs,
            )
            for i in range(0, len(images), batchsize)
        ]
    return subprocess.run([*args, *images, *post_args], check=True, **kwargs)


def ect(
    *images: Union[str, Path],
    level: Optional[int] = None,
    exhaustive: bool = False,
    strip: bool = True,
    strict: bool = True,
    extra_args: Sequence[str] = (),
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
    return batch_run(["ect", *extra_args], *images, check=True)


def optipng(
    *images: Union[str, Path],
    level: int = 5,
    exhaustive: bool = False,
    save_bak: bool = True,
    fix: bool = True,
):
    if not images:
        return
    if level == 5 and exhaustive:
        level = 7
    extra_args = [] if not exhaustive else ["-zm1-9"]
    extra_args += ["-fix"] if fix else []
    remove_bak(*images, save_bak=save_bak)
    return batch_run(["optipng", f"-o{level}", *extra_args], *images, check=True)


def pngcrush(
    *images: Union[str, Path],
    brute: bool = True,
    tmpdir: Optional[Union[str, Path]] = None,
    cleanup: Optional[bool] = None,
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
    args += [str(image) for image in images]

    # Run pngcrush with the specified arguments
    subprocess.run(args, check=True)

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
):
    cur_images = images
    cur_sizes = [Path(image).stat().st_size for image in cur_images]
    while cur_images:
        if shutil.which("ect"):
            for img in tqdm(cur_images, desc="ect"):
                ect(img, exhaustive=exhaustive)
        optipng(*cur_images, exhaustive=exhaustive)
        pngcrush(*cur_images, brute=exhaustive, tmpdir=tmpdir, cleanup=cleanup)
        new_sizes = [Path(image).stat().st_size for image in cur_images]
        cur_images = [
            image
            for image, old_size, new_size in zip(cur_images, cur_sizes, new_sizes)
            if new_size < old_size
        ]
        cur_sizes = [Path(image).stat().st_size for image in cur_images]
