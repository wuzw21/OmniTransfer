"""Deterministic 48D visual evidence for geometric-v9 UI nodes."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any


def deterministic_icon_descriptor_numpy(patches: Any) -> Any:
    """Encode RGB node patches without a learned visual backbone."""

    import numpy as np

    rgb = np.asarray(patches, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("visual patches must have shape [nodes, 3, height, width]")
    if rgb.shape[2] < 4 or rgb.shape[3] < 4:
        raise ValueError("visual patches must be at least 4x4")
    rgb = np.clip(rgb, 0.0, 1.0)
    border = np.concatenate(
        (rgb[:, :, 0, :], rgb[:, :, -1, :], rgb[:, :, :, 0], rgb[:, :, :, -1]),
        axis=2,
    )
    background = border.mean(axis=2, dtype=np.float32)
    color_delta = rgb - background[:, :, None, None]
    absolute_delta = np.abs(color_delta)
    foreground = np.clip(absolute_delta.max(axis=1) * np.float32(4.0), 0.0, 1.0)

    luminance_weights = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    shape = np.abs(
        np.einsum("nchw,c->nhw", color_delta, luminance_weights, optimize=True)
    )
    centered_shape = shape - shape.mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    height, width = shape.shape[1:]
    vertical_basis = _dct_basis(height, 4)
    horizontal_basis = _dct_basis(width, 4)
    dct = np.einsum(
        "fh,nhw,gw->nfg",
        vertical_basis,
        centered_shape,
        horizontal_basis,
        optimize=True,
    ) / np.float32(math.sqrt(height * width))
    dct = dct.reshape(len(rgb), 16).astype(np.float32, copy=False)
    dct[:, 0] = foreground.mean(axis=(1, 2), dtype=np.float32)
    dct *= np.float32(3.0)

    gradient_x = np.zeros_like(shape)
    gradient_y = np.zeros_like(shape)
    gradient_x[:, :, 1:-1] = (shape[:, :, 2:] - shape[:, :, :-2]) * np.float32(0.5)
    gradient_y[:, 1:-1, :] = (shape[:, 2:, :] - shape[:, :-2, :]) * np.float32(0.5)
    absolute_x = np.abs(gradient_x)
    absolute_y = np.abs(gradient_y)
    diagonal = np.minimum(absolute_x, absolute_y) * np.float32(math.sqrt(2.0))
    orientation_energy = np.stack(
        (
            np.maximum(absolute_x - absolute_y, 0.0),
            diagonal,
            np.maximum(absolute_y - absolute_x, 0.0),
        ),
        axis=1,
    )
    hog_cells = []
    for row_slice in _two_slices(height):
        for column_slice in _two_slices(width):
            energy = orientation_energy[:, :, row_slice, column_slice].sum(
                axis=(2, 3), dtype=np.float32
            )
            total_energy = energy.sum(axis=1, keepdims=True)
            normalized_energy = energy / np.maximum(
                total_energy, np.float32(1e-6)
            )
            hog_cells.append(
                np.where(
                    total_energy > np.float32(1e-6),
                    normalized_energy - np.float32(1.0 / 3.0),
                    np.float32(0.0),
                )
            )
    hog = np.concatenate(hog_cells, axis=1) * np.float32(3.0)

    occupancy = np.concatenate(
        (
            _band_means(foreground, axis=2),
            _band_means(foreground, axis=1),
        ),
        axis=1,
    )

    foreground_sum = foreground.sum(axis=(1, 2), keepdims=True, dtype=np.float32)
    safe_sum = np.maximum(foreground_sum, np.float32(1e-6))
    color_mean = (
        (absolute_delta * foreground[:, None]).sum(axis=(2, 3), dtype=np.float32)
        / safe_sum[:, 0]
    )
    centered_color = absolute_delta - color_mean[:, :, None, None]
    color_std = np.sqrt(
        np.maximum(
            (np.square(centered_color) * foreground[:, None]).sum(
                axis=(2, 3), dtype=np.float32
            )
            / safe_sum[:, 0],
            np.float32(0.0),
        )
    )
    fill_ratio = foreground.mean(axis=(1, 2), dtype=np.float32)[:, None]
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    edge_density = np.clip(
        gradient_magnitude.mean(axis=(1, 2), dtype=np.float32) * np.float32(4.0),
        0.0,
        1.0,
    )[:, None]
    color_statistics = np.concatenate(
        (color_mean, color_std, fill_ratio, edge_density), axis=1
    )

    contrast = np.clip(
        shape.mean(axis=(1, 2), dtype=np.float32) * np.float32(4.0), 0.0, 1.0
    )[:, None]
    saturation = (rgb.max(axis=1) - rgb.min(axis=1))
    saturation = (
        (saturation * foreground).sum(axis=(1, 2), dtype=np.float32)
        / safe_sum[:, 0, 0]
    )[:, None]
    row_start, row_end = height // 4, height - height // 4
    column_start, column_end = width // 4, width - width // 4
    center_mass = foreground[:, row_start:row_end, column_start:column_end].sum(
        axis=(1, 2), dtype=np.float32
    )
    centeredness = np.clip(center_mass / safe_sum[:, 0, 0], 0.0, 1.0)[:, None]
    border_foreground = np.concatenate(
        (
            foreground[:, 0, :],
            foreground[:, -1, :],
            foreground[:, :, 0],
            foreground[:, :, -1],
        ),
        axis=1,
    ).mean(axis=1, dtype=np.float32)
    border_cleanliness = (np.float32(1.0) - border_foreground)[:, None]
    quality = np.concatenate(
        (contrast, saturation, centeredness, border_cleanliness), axis=1
    )

    descriptor = np.concatenate(
        (dct, hog, occupancy, color_statistics, quality), axis=1
    ).astype(np.float32, copy=False)
    if descriptor.shape[1] != 48:
        raise AssertionError(f"unexpected visual descriptor width: {descriptor.shape[1]}")
    return descriptor


def multiscale_hash_descriptor_numpy(patches: Any) -> Any:
    """Encode a native-resolution node crop and its context ring separately."""

    import numpy as np

    rgb = np.asarray(patches, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[1] != 6:
        raise ValueError(
            "multiscale visual patches must have shape [nodes, 6, height, width]"
        )
    return np.concatenate(
        (
            deterministic_icon_descriptor_numpy(rgb[:, :3]),
            deterministic_icon_descriptor_numpy(rgb[:, 3:]),
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def deterministic_icon_descriptor_torch(patches: Any, *, torch: Any) -> Any:
    """Torch equivalent of :func:`deterministic_icon_descriptor_numpy`."""

    if patches.ndim != 4 or patches.shape[1] != 3:
        raise ValueError("visual patches must have shape [nodes, 3, height, width]")
    if patches.shape[2] < 4 or patches.shape[3] < 4:
        raise ValueError("visual patches must be at least 4x4")
    rgb = patches.clamp(0.0, 1.0)
    border = torch.cat(
        (rgb[:, :, 0, :], rgb[:, :, -1, :], rgb[:, :, :, 0], rgb[:, :, :, -1]),
        dim=2,
    )
    background = border.mean(dim=2)
    color_delta = rgb - background[:, :, None, None]
    absolute_delta = color_delta.abs()
    foreground = (absolute_delta.amax(dim=1) * 4.0).clamp(0.0, 1.0)

    luminance_weights = torch.as_tensor(
        (0.299, 0.587, 0.114), dtype=rgb.dtype, device=rgb.device
    )
    shape = torch.einsum("nchw,c->nhw", color_delta, luminance_weights).abs()
    centered_shape = shape - shape.mean(dim=(1, 2), keepdim=True)
    height, width = shape.shape[1:]
    vertical_basis = torch.as_tensor(
        _dct_basis(height, 4), dtype=rgb.dtype, device=rgb.device
    )
    horizontal_basis = torch.as_tensor(
        _dct_basis(width, 4), dtype=rgb.dtype, device=rgb.device
    )
    dct = torch.einsum(
        "fh,nhw,gw->nfg", vertical_basis, centered_shape, horizontal_basis
    ) / math.sqrt(height * width)
    dct = dct.reshape(len(rgb), 16)
    dct = torch.cat((foreground.mean(dim=(1, 2), keepdim=False)[:, None], dct[:, 1:]), dim=1)
    dct = dct * 3.0

    gradient_x = torch.zeros_like(shape)
    gradient_y = torch.zeros_like(shape)
    gradient_x[:, :, 1:-1] = (shape[:, :, 2:] - shape[:, :, :-2]) * 0.5
    gradient_y[:, 1:-1, :] = (shape[:, 2:, :] - shape[:, :-2, :]) * 0.5
    absolute_x = gradient_x.abs()
    absolute_y = gradient_y.abs()
    diagonal = torch.minimum(absolute_x, absolute_y) * math.sqrt(2.0)
    orientation_energy = torch.stack(
        (
            torch.clamp_min(absolute_x - absolute_y, 0.0),
            diagonal,
            torch.clamp_min(absolute_y - absolute_x, 0.0),
        ),
        dim=1,
    )
    hog_cells = []
    for row_slice in _two_slices(height):
        for column_slice in _two_slices(width):
            energy = orientation_energy[:, :, row_slice, column_slice].sum(dim=(2, 3))
            total_energy = energy.sum(dim=1, keepdim=True)
            normalized_energy = energy / total_energy.clamp_min(1e-6)
            hog_cells.append(
                torch.where(
                    total_energy > 1e-6,
                    normalized_energy - (1.0 / 3.0),
                    torch.zeros_like(normalized_energy),
                )
            )
    hog = torch.cat(hog_cells, dim=1) * 3.0

    occupancy = torch.cat(
        (
            _torch_band_means(foreground, axis=2, torch=torch),
            _torch_band_means(foreground, axis=1, torch=torch),
        ),
        dim=1,
    )
    foreground_sum = foreground.sum(dim=(1, 2), keepdim=True)
    safe_sum = foreground_sum.clamp_min(1e-6)
    color_mean = (absolute_delta * foreground[:, None]).sum(dim=(2, 3)) / safe_sum[:, 0]
    centered_color = absolute_delta - color_mean[:, :, None, None]
    color_std = torch.sqrt(
        torch.clamp_min(
            (centered_color.square() * foreground[:, None]).sum(dim=(2, 3))
            / safe_sum[:, 0],
            0.0,
        )
    )
    fill_ratio = foreground.mean(dim=(1, 2))[:, None]
    gradient_magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())
    edge_density = (gradient_magnitude.mean(dim=(1, 2)) * 4.0).clamp(0.0, 1.0)[:, None]
    color_statistics = torch.cat(
        (color_mean, color_std, fill_ratio, edge_density), dim=1
    )

    contrast = (shape.mean(dim=(1, 2)) * 4.0).clamp(0.0, 1.0)[:, None]
    saturation_map = rgb.amax(dim=1) - rgb.amin(dim=1)
    saturation = (
        (saturation_map * foreground).sum(dim=(1, 2)) / safe_sum[:, 0, 0]
    )[:, None]
    row_start, row_end = height // 4, height - height // 4
    column_start, column_end = width // 4, width - width // 4
    center_mass = foreground[:, row_start:row_end, column_start:column_end].sum(
        dim=(1, 2)
    )
    centeredness = (center_mass / safe_sum[:, 0, 0]).clamp(0.0, 1.0)[:, None]
    border_foreground = torch.cat(
        (
            foreground[:, 0, :],
            foreground[:, -1, :],
            foreground[:, :, 0],
            foreground[:, :, -1],
        ),
        dim=1,
    ).mean(dim=1)
    border_cleanliness = (1.0 - border_foreground)[:, None]
    quality = torch.cat(
        (contrast, saturation, centeredness, border_cleanliness), dim=1
    )
    descriptor = torch.cat(
        (dct, hog, occupancy, color_statistics, quality), dim=1
    )
    if descriptor.shape[1] != 48:
        raise AssertionError(f"unexpected visual descriptor width: {descriptor.shape[1]}")
    return descriptor


def multiscale_hash_descriptor_torch(patches: Any, *, torch: Any) -> Any:
    """Torch equivalent of :func:`multiscale_hash_descriptor_numpy`."""

    if patches.ndim != 4 or patches.shape[1] != 6:
        raise ValueError(
            "multiscale visual patches must have shape [nodes, 6, height, width]"
        )
    return torch.cat(
        (
            deterministic_icon_descriptor_torch(patches[:, :3], torch=torch),
            deterministic_icon_descriptor_torch(patches[:, 3:], torch=torch),
        ),
        dim=1,
    )


@lru_cache(maxsize=16)
def _dct_basis(size: int, frequencies: int) -> Any:
    import numpy as np

    positions = np.arange(size, dtype=np.float32) + np.float32(0.5)
    frequency = np.arange(frequencies, dtype=np.float32)[:, None]
    basis = np.cos(np.float32(math.pi / size) * frequency * positions[None])
    basis[0] *= np.float32(math.sqrt(1.0 / size))
    basis[1:] *= np.float32(math.sqrt(2.0 / size))
    return basis.astype(np.float32, copy=False)


def _two_slices(size: int) -> tuple[slice, slice]:
    midpoint = size // 2
    return slice(0, midpoint), slice(midpoint, size)


def _band_means(values: Any, *, axis: int) -> Any:
    import numpy as np

    size = values.shape[axis]
    boundaries = [round(index * size / 4) for index in range(5)]
    bands = []
    for index in range(4):
        selector = [slice(None)] * values.ndim
        selector[axis] = slice(boundaries[index], boundaries[index + 1])
        bands.append(values[tuple(selector)].mean(axis=(1, 2), dtype=np.float32))
    return np.stack(bands, axis=1)


def _torch_band_means(values: Any, *, axis: int, torch: Any) -> Any:
    size = values.shape[axis]
    boundaries = [round(index * size / 4) for index in range(5)]
    bands = []
    for index in range(4):
        selector = [slice(None)] * values.ndim
        selector[axis] = slice(boundaries[index], boundaries[index + 1])
        bands.append(values[tuple(selector)].mean(dim=(1, 2)))
    return torch.stack(bands, dim=1)
