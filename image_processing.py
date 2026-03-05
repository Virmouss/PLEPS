import os
import re

import cv2
import numpy as np
import rasterio
from PIL import Image


def is_blank(tile, threshold=0.99):
    """Check if a tile is mostly blank (all white or all black).

    Returns True when the ratio of max-value (255) *or* min-value (0) pixels
    exceeds *threshold*.
    """
    flat_tile = tile.flatten()
    white_ratio = np.sum(flat_tile == 255) / flat_tile.size
    black_ratio = np.sum(flat_tile == 0) / flat_tile.size
    return white_ratio >= threshold or black_ratio >= threshold


def load_and_split_image(image_path, tile_size, output_dir):
    """Load a rasterio-compatible image and split it into tiles.

    Small images (both dimensions < *tile_size*) are resized to *tile_size*
    instead of being tiled.

    Returns
    -------
    list[str]
        Sorted list of tile file paths written to *output_dir*.
    """
    image_name = os.path.basename(image_path)

    with rasterio.open(image_path) as src:
        res_x = src.width
        res_y = src.height

        if res_x < tile_size or res_y < tile_size:
            # Small image – resize to tile_size × tile_size
            img = Image.open(image_path)
            resized_img = img.resize((tile_size, tile_size), Image.BILINEAR)
            resized_img.save(os.path.join(output_dir, f"{image_name}.jpeg"))
            split = False
        else:
            split = True
            for i in range(0, src.height, tile_size):
                for j in range(0, src.width, tile_size):
                    window = rasterio.windows.Window(j, i, tile_size, tile_size)
                    tile = src.read(window=window)

                    cur_h = tile.shape[1]
                    cur_w = tile.shape[2]

                    if cur_h != tile_size or cur_w != tile_size:
                        black_tile = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
                        black_tile[:, :cur_h, :cur_w] = tile[:3, :, :]
                        tile_to_save = black_tile
                    else:
                        tile_to_save = tile

                    if is_blank(tile_to_save[:3]):
                        continue

                    if tile_to_save.shape[0] >= 3:
                        rgb_tile = tile_to_save[:3].transpose(1, 2, 0)
                        image = Image.fromarray(rgb_tile.astype(np.uint8))
                        output_path = os.path.join(output_dir, f"tile_{i}_{j}.jpeg")
                        image.save(output_path, format="JPEG")

    # Collect tile paths
    tiles = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    if split:
        tiles.sort(
            key=lambda x: tuple(
                map(int, re.search(r"tile_(\d+)_(\d+)", os.path.basename(x)).groups())
            )
        )

    return tiles


def draw_detections(image, result):
    """Draw detection bounding boxes on *image* (in-place) and return it."""
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image


def stitch_tiles(tiles, tile_results, project_name, draw_fn):
    """Stitch processed tiles back into a single image and save it.

    Parameters
    ----------
    tiles : list[str]
        Ordered tile file paths.
    tile_results : list
        YOLO result objects aligned with *tiles*.
    project_name : str
        Used to name the output file.
    draw_fn : callable
        Function ``(image, result) -> image`` used to overlay detections.
    """
    pattern = r"tile_(\d+)_(\d+)\.jpeg"
    tile_info = []

    for filename in tiles:
        match = re.search(pattern, filename)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            tile_info.append((row, col, filename))

    tile_info.sort(key=lambda x: (x[0], x[1]))

    rows = sorted({info[0] for info in tile_info})
    cols = sorted({info[1] for info in tile_info})

    print(f"Found {len(tile_info)} tiles")
    print(f"Grid dimensions: {len(rows)} rows × {len(cols)} columns")

    first_tile = cv2.imread(tile_info[0][2])
    if first_tile is None:
        print(f"Error: Could not load {tile_info[0][2]}")
        return

    tile_height, tile_width = first_tile.shape[:2]
    output_width = max(cols) + tile_width
    output_height = max(rows) + tile_height

    print(f"Output image size: {output_width} × {output_height} pixels")

    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for idx, (row, col, filename) in enumerate(tile_info):
        tile = cv2.imread(filename)
        result = tile_results[idx]
        tile = draw_fn(tile, result)

        tile_h, tile_w = tile.shape[:2]
        output_image[row : row + tile_h, col : col + tile_w] = tile

    output_path = f"{project_name}_complete.jpeg"
    cv2.imwrite(output_path, output_image)
    print(f"Saved stitched image to {output_path}")
