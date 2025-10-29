"""Standalone utilities for representing ARC-style grids as text, diffs, and base64 PNGs."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

GRID = list[list[int]]
"""Alias used throughout the module to describe a 2D grid of integer color indices."""

DEFAULT_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # black
    1: (0, 0, 255),  # blue
    2: (255, 0, 0),  # red
    3: (0, 255, 0),  # green
    4: (255, 255, 0),  # yellow
    5: (192, 192, 192),  # gray
    6: (255, 0, 255),  # magenta
    7: (255, 165, 0),  # orange
    8: (0, 255, 255),  # cyan
    9: (128, 255, 128),  # light green
}


@dataclass(slots=True, frozen=True)
class GridShape:
    rows: int
    cols: int


def grid_to_text(
    grid: GRID,
    *,
    cell_separator: str = " ",
    row_separator: str = "\n",
) -> str:
    """Render a grid as the whitespace-separated string format used when prompting LLMs.

    Args:
        grid: 2D list of integers.
        cell_separator: String placed between cells in a row.
        row_separator: String placed between rows.

    Returns:
        String representation mirroring [`Challenge.grid_to_str()`](src/models.py:141).

    Raises:
        ValueError: If the grid is empty or ragged.
    """
    _validate_rectangular_grid(grid)
    return row_separator.join(
        cell_separator.join(str(value) for value in row) for row in grid
    )


def text_to_grid(
    text: str,
    *,
    cell_separator: str | None = " ",
    row_separator: str | None = "\n",
) -> GRID:
    """Parse a textual grid representation back into a GRID structure.

    Args:
        text: String containing numeric rows, typically produced by
            [`grid_to_text`](representation_utils.py:27).
        cell_separator: Delimiter between cells inside a row. When ``None``,
            any contiguous whitespace is treated as a separator.
        row_separator: Delimiter between rows. When ``None``, line breaks are used.

    Returns:
        Reconstructed grid as ``list[list[int]]``.

    Raises:
        ValueError: If the text is empty, contains non-integer tokens, or forms a ragged grid.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Cannot parse grid from empty text.")

    if row_separator is None:
        raw_rows = stripped.splitlines()
    else:
        raw_rows = stripped.split(row_separator)

    rows: list[list[int]] = []
    for row_index, raw_row in enumerate(raw_rows):
        row_text = raw_row.strip()
        if not row_text:
            continue

        if cell_separator is None:
            cell_tokens = row_text.split()
        else:
            cell_tokens = [tok for tok in row_text.split(cell_separator) if tok]

        if not cell_tokens:
            raise ValueError(f"Row {row_index} is empty after splitting.")

        try:
            row_values = [int(token) for token in cell_tokens]
        except ValueError as exc:
            raise ValueError(
                f"Row {row_index} contains a non-integer token: {cell_tokens}"
            ) from exc

        rows.append(row_values)

    shape = _validate_rectangular_grid(rows)
    if shape.rows == 0 or shape.cols == 0:
        raise ValueError("Parsed grid has no cells.")

    return rows


def generate_grid_diff(expected_grid: GRID, actual_grid: GRID) -> str:
    """Create the ASCII diff notation used in [`generate_grid_diff()`](src/run.py:211).

    Each cell is rendered as either ``✓value`` when it matches the expected value or
    ``actual→expected`` when it differs. Mismatched row lengths surface as explicit errors.

    Args:
        expected_grid: Ground truth grid.
        actual_grid: Grid produced by a model or heuristic.

    Returns:
        Formatted multi-line string describing differences.

    Raises:
        ValueError: If either grid is empty.
    """
    expected_shape = _validate_rectangular_grid(expected_grid)
    actual_shape = _validate_rectangular_grid(actual_grid)

    if expected_shape != actual_shape:
        return (
            "Error: Grid dimension mismatch "
            f"(rows: {expected_shape.rows}×{expected_shape.cols} "
            f"vs {actual_shape.rows}×{actual_shape.cols})"
        )

    max_width = 0
    for expected_row, actual_row in zip(expected_grid, actual_grid, strict=True):
        for expected_val, actual_val in zip(expected_row, actual_row, strict=True):
            cell_text = (
                f"✓{expected_val}"
                if expected_val == actual_val
                else f"{actual_val}→{expected_val}"
            )
            max_width = max(max_width, len(cell_text))

    max_width += 2  # Left/right padding
    num_cols = expected_shape.cols
    border = "+" + "+".join("-" * max_width for _ in range(num_cols)) + "+"

    diff_lines: list[str] = [border]
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected_grid, actual_grid, strict=True)
    ):
        if len(expected_row) != len(actual_row):
            diff_lines.append(f"| Row {row_index}: Error - column count mismatch |")
            diff_lines.append(border)
            continue

        row_cells: list[str] = []
        for expected_val, actual_val in zip(expected_row, actual_row, strict=True):
            cell_text = (
                f"✓{expected_val}"
                if expected_val == actual_val
                else f"{actual_val}→{expected_val}"
            )
            row_cells.append(cell_text.center(max_width))
        diff_lines.append("|" + "|".join(row_cells) + "|")
        diff_lines.append(border)

    return "\n".join(diff_lines)


def grid_to_base64_png(
    grid: GRID,
    *,
    color_map: dict[int, tuple[int, int, int]] | None = None,
    cell_size: int = 32,
    draw_grid_lines: bool = True,
    grid_line_color: tuple[int, int, int] = (255, 255, 255),
    grid_line_width: int = 1,
) -> str:
    """Convert a grid into a base64-encoded PNG using a lightweight Pillow renderer.

    Args:
        grid: 2D list of integer color indices.
        color_map: Optional mapping from integers to RGB tuples. Missing keys fall back
            to ``DEFAULT_COLOR_MAP`` or to a deterministic gray shade.
        cell_size: Pixel size of each cell edge.
        draw_grid_lines: Whether to draw thin separators between cells.
        grid_line_color: RGB color of the separators.
        grid_line_width: Line width (in pixels) for separators.

    Returns:
        Base64 string ready to embed as ``data:image/png;base64,<value>``.

    Raises:
        ValueError: If the grid is empty or ragged.
        ImportError: If Pillow is not installed in the environment.
    """
    shape = _validate_rectangular_grid(grid)
    resolved_color_map = _build_color_map(color_map)
    image_width = shape.cols * cell_size
    image_height = shape.rows * cell_size

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError(
            "grid_to_base64_png requires the Pillow package. "
            "Install it via `pip install pillow`."
        ) from exc

    image = Image.new("RGB", (image_width, image_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)

    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            rgb = resolved_color_map.get(value, _fallback_color(value))
            top_left = (col_index * cell_size, row_index * cell_size)
            bottom_right = (
                top_left[0] + cell_size,
                top_left[1] + cell_size,
            )
            draw.rectangle([top_left, bottom_right], fill=rgb)

    if draw_grid_lines and grid_line_width > 0:
        for row_index in range(1, shape.rows):
            y = row_index * cell_size
            draw.line(
                [(0, y), (image_width, y)],
                fill=grid_line_color,
                width=grid_line_width,
            )
        for col_index in range(1, shape.cols):
            x = col_index * cell_size
            draw.line(
                [(x, 0), (x, image_height)],
                fill=grid_line_color,
                width=grid_line_width,
            )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_bytes = base64.b64encode(buffer.getvalue())
    return base64_bytes.decode("utf-8")


def _validate_rectangular_grid(grid: GRID) -> GridShape:
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty and rectangular.")

    row_length = len(grid[0])
    for row_index, row in enumerate(grid):
        if len(row) != row_length:
            raise ValueError(
                f"Grid is ragged: row {row_index} has length {len(row)} "
                f"instead of {row_length}."
            )
    return GridShape(rows=len(grid), cols=row_length)


def _build_color_map(
    color_map: dict[int, tuple[int, int, int]] | None,
) -> dict[int, tuple[int, int, int]]:
    combined = dict(DEFAULT_COLOR_MAP)
    if color_map:
        combined.update(color_map)
    return combined


def _fallback_color(value: int) -> tuple[int, int, int]:
    normalized = abs(value) % 256
    return (normalized, normalized, normalized)
