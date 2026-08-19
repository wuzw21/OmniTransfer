from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from omnitransfer.learned_matcher import _visual_inputs as torch_visual_inputs
from omnitransfer.learned_matcher import MULTISCALE_HASH_VISUAL_ENCODER
from omnitransfer.numpy_v9_matcher import _visual_inputs as numpy_visual_inputs
from omnitransfer.ui_graph import UIGraph, UINode


def _graph(
    screenshot: Path,
    *,
    bbox: tuple[float, float, float, float],
    visual_bbox: tuple[float, float, float, float],
) -> UIGraph:
    return UIGraph(
        graph_id="retina-screen",
        width=414.0,
        height=736.0,
        nodes=(
            UINode(
                node_id="icon",
                origin_id="icon",
                bbox=bbox,
                metadata={
                    "visual_bbox": visual_bbox,
                    "visual_bbox_coordinate_space": "page_pixels",
                },
            ),
        ),
        metadata={
            "screenshot_path": str(screenshot),
            "visual_display_size": (1242.0, 2208.0),
        },
    )


def test_retina_visual_bbox_uses_original_screenshot_coordinates(
    tmp_path: Path,
) -> None:
    screenshot = tmp_path / "retina.png"
    image = Image.new("RGB", (1242, 2208), "black")
    ImageDraw.Draw(image).rectangle((540, 1830, 720, 2010), fill=(255, 0, 0))
    image.save(screenshot)
    graph = _graph(
        screenshot,
        bbox=(180.0, 610.0, 240.0, 670.0),
        visual_bbox=(540.0, 1830.0, 720.0, 2010.0),
    )

    numpy_patches, numpy_mask = numpy_visual_inputs(
        graph,
        patch_size=32,
        canvas_size=384,
    )

    assert float(numpy_mask[0, 0]) == 1.0
    assert float(numpy_patches[0, 0].mean()) > 0.8
    assert float(numpy_patches[0, 1:].mean()) < 0.1


def test_explicit_out_of_bounds_visual_bbox_is_unavailable(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (1242, 2208), "white").save(screenshot)
    graph = _graph(
        screenshot,
        bbox=(10.0, 10.0, 30.0, 30.0),
        visual_bbox=(1300.0, 10.0, 1400.0, 30.0),
    )

    _, numpy_mask = numpy_visual_inputs(
        graph,
        patch_size=32,
        canvas_size=384,
    )

    assert float(numpy_mask[0, 0]) == 0.0


def test_torch_and_numpy_visual_inputs_share_retina_coordinates(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    screenshot = tmp_path / "retina.png"
    image = Image.new("RGB", (1242, 2208), "black")
    ImageDraw.Draw(image).rectangle((540, 1830, 720, 2010), fill=(255, 0, 0))
    image.save(screenshot)
    graph = _graph(
        screenshot,
        bbox=(180.0, 610.0, 240.0, 670.0),
        visual_bbox=(540.0, 1830.0, 720.0, 2010.0),
    )

    torch_patches, torch_mask = torch_visual_inputs(
        graph,
        patch_size=32,
        canvas_size=384,
        torch=torch,
        device="cpu",
    )
    numpy_patches, numpy_mask = numpy_visual_inputs(
        graph,
        patch_size=32,
        canvas_size=384,
    )

    assert float(torch_mask[0, 0]) == 1.0
    assert float(numpy_mask[0, 0]) == 1.0
    np.testing.assert_allclose(
        torch_patches[0].numpy(), numpy_patches[0], rtol=0.0, atol=0.12
    )


def test_multiscale_visual_input_keeps_native_icon_and_context_ring(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    screenshot = tmp_path / "small-icon.png"
    image = Image.new("RGB", (1080, 2400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((500, 1000, 543, 1043), fill="black")
    draw.rectangle((552, 1000, 571, 1019), fill="red")
    image.save(screenshot)
    graph = UIGraph(
        graph_id="small-icon",
        width=1080.0,
        height=2400.0,
        nodes=(
            UINode(
                node_id="icon",
                origin_id="icon",
                bbox=(500.0, 1000.0, 544.0, 1044.0),
                metadata={"visual_bbox": (500.0, 1000.0, 544.0, 1044.0)},
            ),
        ),
        metadata={"screenshot_path": str(screenshot)},
    )

    patches, mask = numpy_visual_inputs(
        graph,
        patch_size=32,
        canvas_size=0,
        visual_encoder=MULTISCALE_HASH_VISUAL_ENCODER,
        context_scale=3.0,
    )

    assert patches.shape == (1, 6, 32, 32)
    assert float(mask[0, 0]) == 1.0
    tight, context = patches[0, :3], patches[0, 3:]
    assert float(abs(tight[0].mean() - tight[1].mean())) < 0.01
    assert float(context[0].mean() - context[1].mean()) > 0.01

    torch_patches, torch_mask = torch_visual_inputs(
        graph,
        patch_size=32,
        canvas_size=0,
        visual_encoder=MULTISCALE_HASH_VISUAL_ENCODER,
        context_scale=3.0,
        torch=torch,
        device="cpu",
    )
    assert tuple(torch_patches.shape) == (1, 6, 32, 32)
    assert float(torch_mask[0, 0]) == 1.0
    np.testing.assert_allclose(
        torch_patches[0].numpy(), patches[0], rtol=0.0, atol=0.04
    )
