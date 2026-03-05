import os

import cv2
import pandas as pd
from tkinter import filedialog

from image_processing import draw_detections


def save_project_csv(tile_results, project_name, model_name, image_name,
                     total_tiles, total_objects, inference_time,
                     plantation_area, production):
    """Build per-tile CSV data and save via a file-save dialog.

    Parameters
    ----------
    tile_results : list
        YOLO result objects (one per tile).
    project_name, model_name, image_name : str
        Metadata fields written to every row.
    total_tiles, total_objects : int
        Summary counts.
    inference_time : float
        Total inference time in seconds.
    plantation_area : float
        Area in hectares.
    production : float
        Estimated production value.
    """
    csv_data = []
    for i, result in enumerate(tile_results):
        count_object = len(result.boxes)
        csv_data.append({
            "project_name": project_name,
            "model_name": model_name,
            "file_name": image_name,
            "total_tiles": total_tiles,
            "total_pohon": total_objects,
            "inference_time": inference_time,
            "plantation_area": plantation_area,
            "production_est": production,
            "tile": f"Tile {i + 1}",
            "objects_in_tile": count_object,
        })

    df = pd.DataFrame(csv_data)

    file_path = filedialog.asksaveasfile(
        defaultextension=".csv",
        title="Save Project Results",
        filetypes=[("CSV File", "*.csv")],
    )

    if file_path is not None:
        df.to_csv(file_path, index=False)


def save_detection_images(tiles, tile_results, project_name):
    """Save each tile with drawn detections to disk.

    Images are written to ``saved_images/<project_name>/``.
    """
    output_dir = os.path.join("saved_images", project_name)
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(tiles):
        tile = cv2.imread(img_path)
        display_img = tile.copy()

        result = tile_results[i]
        drawn_img = draw_detections(display_img, result)

        output_path = os.path.join(output_dir, f"Tile_{i + 1}.jpg")
        cv2.imwrite(output_path, drawn_img)
