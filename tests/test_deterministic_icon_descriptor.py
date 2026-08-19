from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from omnitransfer.visual_descriptor import (
    deterministic_icon_descriptor_numpy,
    multiscale_hash_descriptor_numpy,
)


def _patch(kind: str, *, inverted: bool = False) -> np.ndarray:
    background = 0 if inverted else 255
    foreground = 255 if inverted else 0
    image = Image.new("RGB", (32, 32), (background,) * 3)
    draw = ImageDraw.Draw(image)
    if kind == "solid_star":
        draw.regular_polygon((16, 16, 11), 5, rotation=-18, fill=(foreground,) * 3)
    elif kind == "outline_star":
        draw.regular_polygon(
            (16, 16, 11),
            5,
            rotation=-18,
            outline=(foreground,) * 3,
            width=2,
        )
    elif kind == "plus":
        draw.rectangle((13, 5, 19, 27), fill=(foreground,) * 3)
        draw.rectangle((5, 13, 27, 19), fill=(foreground,) * 3)
    else:
        raise ValueError(kind)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.dot(first, second)
        / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12)
    )


def test_deterministic_icon_descriptor_has_fixed_48d_contract() -> None:
    descriptors = deterministic_icon_descriptor_numpy(
        np.stack([_patch("solid_star"), _patch("outline_star"), _patch("plus")])
    )

    assert descriptors.shape == (3, 48)
    assert descriptors.dtype == np.float32
    assert np.isfinite(descriptors).all()


def test_icon_descriptor_preserves_state_but_separates_shape() -> None:
    solid, outline, plus = deterministic_icon_descriptor_numpy(
        np.stack([_patch("solid_star"), _patch("outline_star"), _patch("plus")])
    )

    state_similarity = _cosine(solid, outline)
    wrong_shape_similarity = _cosine(solid, plus)
    assert 0.25 < state_similarity < 0.98
    assert state_similarity > wrong_shape_similarity + 0.08


def test_icon_descriptor_is_robust_to_light_dark_theme_inversion() -> None:
    light, dark = deterministic_icon_descriptor_numpy(
        np.stack([_patch("outline_star"), _patch("outline_star", inverted=True)])
    )

    assert _cosine(light, dark) > 0.85


def test_multiscale_hash_keeps_tight_shape_and_context_separate() -> None:
    tight_star = _patch("outline_star")
    tight_plus = _patch("plus")
    context = _patch("solid_star")
    descriptors = multiscale_hash_descriptor_numpy(
        np.stack(
            [
                np.concatenate((tight_star, context), axis=0),
                np.concatenate((tight_plus, context), axis=0),
            ]
        )
    )

    assert descriptors.shape == (2, 96)
    np.testing.assert_allclose(descriptors[0, 48:], descriptors[1, 48:])
    assert _cosine(descriptors[0, :48], descriptors[1, :48]) < 0.9
