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
