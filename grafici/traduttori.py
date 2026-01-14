import io
from kivy.graphics.texture import Texture


def traduci(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    buf.close()
    texture = Texture.create(size=(1, 1))
    texture.blit_buffer(data, colorfmt="rgba", bufferfmt="ubyte")
    texture.flip_vertical()
    return texture
