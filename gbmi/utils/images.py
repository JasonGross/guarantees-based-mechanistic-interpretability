from typing import Union, Optional
import tempfile
import subprocess
from pathlib import Path
from PIL import Image, ImageChops
import io
import plotly.graph_objects as go


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


def optipng(*images: Union[str, Path], level: int = 5):
    if not images:
        return
    return subprocess.run(["optipng", f"-o{level}", *images], check=True)


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
