import av
import pytest
from PIL import Image, ImageDraw


def _gray(value: int, w: int = 64, h: int = 48) -> Image.Image:
    return Image.new("RGB", (w, h), (value, value, value))


@pytest.fixture
def make_video(tmp_path):
    """Write a short video whose frame k is a flat gray of value 8*k.

    Flat gray survives any codec, so a decoded frame identifies its own index.
    """

    def _make(name: str = "clip.mp4", frames: int = 30, fps: int = 10,
              w: int = 64, h: int = 48):
        path = tmp_path / name
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            for k in range(frames):
                frame = av.VideoFrame.from_image(_gray(8 * k, w, h))
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        return path

    return _make


@pytest.fixture
def workspace(tmp_path):
    """A folder of images with known sizes, plus one duplicate."""
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    Image.new("RGB", (4032, 3024), (10, 20, 30)).save(root / "pallet_01.jpg")
    Image.new("RGB", (800, 600), (40, 50, 60)).save(root / "pallet_02.png")
    Image.new("RGB", (800, 600), (40, 50, 60)).save(root / "sub" / "copy_of_02.png")
    (root / "notes.txt").write_text("not an asset")
    return root


MARKER_BOX = (3000, 2000, 3024, 2024)  # source pixels; centre (3011.5, 2011.5)


@pytest.fixture
def marker_workspace(tmp_path):
    """One big image with a red square at known source pixels, plus a video.

    Everything about `view` is a coordinate claim, so the tests need a target whose
    source position is known independently of the code under test.
    """
    root = tmp_path / "mws"
    root.mkdir()
    img = Image.new("RGB", (4032, 3024), (30, 30, 30))
    ImageDraw.Draw(img).rectangle([MARKER_BOX[0], MARKER_BOX[1],
                                   MARKER_BOX[2] - 1, MARKER_BOX[3] - 1], fill=(255, 0, 0))
    img.save(root / "marker.png")

    with av.open(str(root / "clip.mp4"), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for k in range(90):
            frame = av.VideoFrame.from_image(Image.new("RGB", (320, 240), (2 * k,) * 3))
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return root


CORPUS_SIZE = 1000
CORPUS_SIZES = [(1024, 768), (800, 600), (1280, 720), (960, 960)]


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """A thousand-image folder, indexed. The scale the Phase 1 exit asks for.

    Deliberately synthetic: the exit criterion is about the harness holding, not
    about what is in the pictures, and a fixture that needs a real dataset on
    disk cannot run anywhere but one laptop. Sizes vary so `fit_render_size`
    takes more than one path through its loop.

    Session-scoped because generating and hashing a thousand files is a few
    seconds, and no test mutates the source images.
    """
    root = tmp_path_factory.mktemp("corpus")
    for i in range(CORPUS_SIZE):
        w, h = CORPUS_SIZES[i % len(CORPUS_SIZES)]
        img = Image.new("RGB", (w, h), ((i * 7) % 256, (i * 13) % 256, (i * 29) % 256))
        draw = ImageDraw.Draw(img)
        # Something small and off-centre, so a zoomed view differs from a wide one.
        draw.rectangle([w // 4, h // 4, w * 3 // 4, h * 3 // 4], fill=(255, 40, 40))
        draw.text((8, 8), f"plate {i:04d}", fill=(255, 255, 255))
        img.save(root / f"plate_{i:04d}.png")

    from visru.workspace.registry import Registry

    reg = Registry(root)
    reg.scan()
    reg.save()
    return root


def pytest_addoption(parser):
    parser.addoption("--run-models", action="store_true",
                     help="run tests that download and execute real model weights")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-models"):
        return
    skip = pytest.mark.skip(reason="needs real weights; run with --run-models")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)
