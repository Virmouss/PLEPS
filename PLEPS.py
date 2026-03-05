import os
import shutil
import time
import threading

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image
import cv2
import numpy as np

from ultralytics import YOLO

from dialogs import (
    InitWindow,
    ModelSelectWindow,
    DetectParamWindow,
    CalcSettingWindow,
    AboutWindow,
    UserManualWindow,
)
from image_processing import (
    load_and_split_image,
    draw_detections,
    stitch_tiles,
)
from file_io import save_project_csv, save_detection_images


# Main Application
class YOLOv8TiledApp(ctk.CTk):

    # Initialization
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("PLEPS")
        self.geometry("1400x900")
        self._init_state()

    def _init_state(self):
        """Reset / initialize all application state."""
        self.image_output_dir = "split_images"
        os.makedirs(self.image_output_dir, exist_ok=True)

        self.project_name = ""
        self._prompt_project_name()

        # Model
        self.model = YOLO("models\\yolov8m.pt")
        self.tile_size = 1000
        self.overlap = 0.2

        # Image & results
        self.image_path = None
        self.image_name = None
        self.original_image = None
        self.displayed_image = None
        self.tiles = []
        self.tile_results = []
        self.current_tile_index = 0
        self.total_objects = 0
        self.inference_time = 0

        # Dialog tracking
        self.toplevel_window = None
        self.model_selected_option = {"value": "", "text": ""}
        self.model_custom = "Yolov8m"
        self.model_detect_param = {"conf": 50, "iou": 45}
        self.prod_calc_var = {"area": 100, "tbs": 0.13}
        self.production = 0.0

        self.csv_data = []

        # Build the UI
        self._create_layout()

    def _prompt_project_name(self):
        init = InitWindow(self, text="Enter Project Name")
        self.project_name = init.get_input()

    # Layout
    def _create_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(1, weight=3)
        main_frame.grid_rowconfigure(0, weight=1)

        self._create_menu()

        # --- Left sidebar (scrollable) ---
        left_frame = ctk.CTkScrollableFrame(
            main_frame, width=380, height=550,
            scrollbar_fg_color="transparent",
            scrollbar_button_color="#AAAAAA",
            scrollbar_button_hover_color="#888888",
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_columnconfigure(0, weight=1)

        # --- Right image panel ---
        self.image_frame = ctk.CTkFrame(main_frame)
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(1, weight=1)

        self.image_title = ctk.CTkLabel(
            self.image_frame, text="Detection Results",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.image_title.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self._create_tile_nav_widgets()

        # --- Left sidebar sections ---
        self._create_control_section(left_frame)
        self._create_project_section(left_frame)
        self._create_metrics_section(left_frame)
        self._create_navigation_section(left_frame)
        self._create_production_section(left_frame)

        # --- Image display area ---
        self.image_display_frame = ctk.CTkScrollableFrame(
            self.image_frame, fg_color="transparent",
        )
        self.image_display_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.image_display_frame.grid_columnconfigure(0, weight=1)
        self.image_display_frame.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.image_display_frame, text="")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # --- Status bar ---
        status_frame = ctk.CTkFrame(self.image_frame)
        status_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        status_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(status_frame, text="Ready")
        self.status_label.grid(row=0, column=1, padx=10)

    # Menu bar
    def _create_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self._new_project)
        file_menu.add_command(label="Save", command=self._save_project)
        file_menu.add_command(label="Save Image Detections", command=self._save_image_detects)
        file_menu.add_command(label="Stitch", command=self._stitch_tiles)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)

        setting_menu = tk.Menu(menubar, tearoff=0)
        setting_menu.add_command(label="Model", command=self._open_model_select)
        setting_menu.add_command(label="Detection Parameter", command=self._open_detect_params)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Manual", command=self._open_user_manual)
        help_menu.add_command(label="About", command=self._open_about)

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Setting", menu=setting_menu)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menubar)

    # Widget creation helpers
    def _create_tile_nav_widgets(self):
        nav_control_frame = ctk.CTkFrame(self.image_frame)
        nav_control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        nav_control_frame.grid_columnconfigure((0, 1), weight=1)

        self.prev_btn = ctk.CTkButton(
            nav_control_frame, text="◀ Previous",
            command=self._show_previous_tile, state="disabled",
        )
        self.prev_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.next_btn = ctk.CTkButton(
            nav_control_frame, text="Next ▶",
            command=self._show_next_tile, state="disabled",
        )
        self.next_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        current_tile_frame = ctk.CTkFrame(self.image_frame)
        current_tile_frame.grid(row=3, column=0, sticky="w", padx=10, pady=5)
        current_tile_frame.grid_columnconfigure(0, weight=1)

        self.current_tile_label = ctk.CTkLabel(current_tile_frame, text="Current tile")
        self.current_tile_label.grid(row=0, column=0, sticky="e", padx=5, pady=2)

        self.current_tile_label_value = ctk.CTkLabel(current_tile_frame, text=" 0/0")
        self.current_tile_label_value.grid(row=0, column=1, sticky="w", padx=5)

        self.objects_count_label = ctk.CTkLabel(current_tile_frame, text="Objects in tile")
        self.objects_count_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)

        self.objects_count_label_value = ctk.CTkLabel(current_tile_frame, text=" 0")
        self.objects_count_label_value.grid(row=1, column=1, sticky="w", padx=5)

    def _create_control_section(self, parent):
        control_section = ctk.CTkFrame(parent)
        control_section.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        control_section.grid_columnconfigure((0, 1), weight=1)

        load_btn = ctk.CTkButton(control_section, text="Load Image", command=self._load_image)
        load_btn.grid(row=0, column=0, padx=3, pady=5, sticky="e")

        process_btn = ctk.CTkButton(control_section, text="Process Image", command=self._process_image)
        process_btn.grid(row=0, column=1, padx=3, pady=5, sticky="w")

    def _create_project_section(self, parent):
        self.project_frame = ctk.CTkFrame(parent)
        self.project_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(
            self.project_frame, text="Project Details",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        VALUE_FONT = ctk.CTkFont(weight="bold")
        VALUE_COLOR = "#5e6f3b"

        ctk.CTkLabel(self.project_frame, text="Project Name :").grid(
            row=1, column=0, sticky="e", padx=2, pady=2)
        self.project_name_label_value = ctk.CTkLabel(
            self.project_frame, text=f"{self.project_name}",
            font=VALUE_FONT, text_color=VALUE_COLOR,
        )
        self.project_name_label_value.grid(row=1, column=1, sticky="w", padx=2, pady=2)

        ctk.CTkLabel(self.project_frame, text="Image File :").grid(
            row=2, column=0, sticky="e", padx=2, pady=2)
        self.image_file_label_value = ctk.CTkLabel(
            self.project_frame, text=" -",
            font=VALUE_FONT, text_color=VALUE_COLOR,
        )
        self.image_file_label_value.grid(row=2, column=1, sticky="w", padx=2, pady=2)

    def _create_metrics_section(self, parent):
        self.metrics_frame = ctk.CTkFrame(parent)
        self.metrics_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(
            self.metrics_frame, text="Detection Metrics",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        VALUE_FONT = ctk.CTkFont(weight="bold")
        VALUE_COLOR = "#5e6f3b"

        # Model
        ctk.CTkLabel(self.metrics_frame, text="Model :").grid(
            row=1, column=0, sticky="e", padx=2, pady=2)
        self.model_status_label_value = ctk.CTkLabel(
            self.metrics_frame, text=" Yolov8m",
            font=VALUE_FONT, text_color=VALUE_COLOR,
        )
        self.model_status_label_value.grid(row=1, column=1, sticky="w", padx=2, pady=2)

        # Total tiles
        ctk.CTkLabel(self.metrics_frame, text="Total tiles :").grid(
            row=2, column=0, sticky="e", padx=2, pady=2)
        self.tile_count_label_value = ctk.CTkLabel(
            self.metrics_frame, text=" 0",
            font=VALUE_FONT, text_color=VALUE_COLOR,
        )
        self.tile_count_label_value.grid(row=2, column=1, sticky="w", padx=2, pady=2)

        # Total objects
        ctk.CTkLabel(self.metrics_frame, text="Total objects :").grid(
            row=3, column=0, sticky="e", padx=2, pady=2)
        self.total_objects_label_value = ctk.CTkLabel(
            self.metrics_frame, text=" 0",
            font=VALUE_FONT, text_color=VALUE_COLOR,
        )
        self.total_objects_label_value.grid(row=3, column=1, sticky="w", padx=2, pady=2)

        # Inference time
        ctk.CTkLabel(self.metrics_frame, text="Total inference time :").grid(
            row=4, column=0, sticky="e", padx=2, pady=2)
        self.inference_time_label_value = ctk.CTkLabel(
            self.metrics_frame, text=" -",
            font=VALUE_FONT, text_color=VALUE_COLOR,
        )
        self.inference_time_label_value.grid(row=4, column=1, sticky="w", padx=2, pady=2)

    def _create_navigation_section(self, parent):
        self.nav_frame = ctk.CTkFrame(parent)
        self.nav_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(
            self.nav_frame, text="Tile Navigation",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.tile_listbox_frame = ctk.CTkFrame(self.nav_frame)
        self.tile_listbox_frame.grid(row=2, column=0, sticky="nsew", padx=100, pady=5)
        self.tile_listbox_frame.grid_columnconfigure(0, weight=1)
        self.tile_listbox_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.tile_listbox_frame, text="Tile List:").grid(
            row=0, column=0, sticky="w", pady=2)

        listbox_frame = ctk.CTkFrame(self.tile_listbox_frame)
        listbox_frame.grid(row=1, column=0, sticky="ew", padx=10)
        listbox_frame.grid_columnconfigure(0, weight=1)
        listbox_frame.grid_rowconfigure(0, weight=1)

        self.tile_listbox = tk.Listbox(
            listbox_frame, bg="#2b2b2b", fg="white",
            selectbackground="#1f538d", selectforeground="white",
        )
        self.tile_listbox.grid(row=0, column=0, sticky="ew", padx=10)

        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.tile_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tile_listbox.configure(yscrollcommand=scrollbar.set)

        self.tile_listbox.bind("<<ListboxSelect>>", self._on_tile_selected)

    def _create_production_section(self, parent):
        self.results_frame = ctk.CTkFrame(parent)
        self.results_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkLabel(
            self.results_frame, text="Palm Oil Production Estimation",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        calc_btn = ctk.CTkButton(
            self.results_frame, text="Calculation Settings",
            command=self._open_prod_calc,
        )
        calc_btn.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.result_area_label = ctk.CTkLabel(self.results_frame, text="Plantation Area: -")
        self.result_area_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)

        self.result_production_label = ctk.CTkLabel(
            self.results_frame, text="Estimated Plantation Oil Palm Production: -",
        )
        self.result_production_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)

    # Dialog openers
    def _open_toplevel(self, window_cls, *args, wait=False):
        """Open a toplevel window, reusing the existing one if still open."""
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = window_cls(self, *args)
            self.toplevel_window.after(100, self.toplevel_window.lift)
            if wait:
                self.wait_window(self.toplevel_window)
        else:
            self.toplevel_window.focus()

    def _open_model_select(self):
        self._open_toplevel(ModelSelectWindow, self.model_selected_option)

    def _open_detect_params(self):
        self._open_toplevel(DetectParamWindow, self.model_detect_param)

    def _open_prod_calc(self):
        self._open_toplevel(CalcSettingWindow, self.prod_calc_var, wait=True)
        self._update_production_calc()

    def _open_about(self):
        self._open_toplevel(AboutWindow, wait=True)

    def _open_user_manual(self):
        self._open_toplevel(UserManualWindow, wait=True)

    # Image loading
    def _load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )

        if self.image_path != "":
            self.tiles = []
            shutil.rmtree("split_images", ignore_errors=True)
            os.makedirs(self.image_output_dir, exist_ok=True)

            image_name = os.path.basename(self.image_path)
            self.image_file_label_value.configure(text=f" {image_name}")
            self.image_name = image_name

            self.tiles = load_and_split_image(
                self.image_path, self.tile_size, self.image_output_dir,
            )

            self._show_current_tile()

    # Image display
    def _display_image(self, image):
        """Display *image* (numpy array or file path) on the main label."""
        if self.tile_results and self.tile_results[self.current_tile_index] is not None:
            # image is an OpenCV numpy array with detections drawn
            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height() - 200

            if frame_width <= 1 or frame_height <= 1:
                frame_width, frame_height = 800, 600

            h, w = image.shape[:2]
            scale = min(frame_width / w, frame_height / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(image, (new_w, new_h))

            if len(resized.shape) == 3 and resized.shape[2] == 3:
                if isinstance(image, np.ndarray):
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            self.displayed_image = ctk.CTkImage(Image.fromarray(resized), size=(new_w, new_h))
            self.image_label.configure(image=self.displayed_image)
        else:
            # image is a file path string
            img = Image.open(image)

            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height() - 200
            if frame_width <= 1 or frame_height <= 1:
                frame_width, frame_height = 800, 600

            w, h = img.size
            scale = min(frame_width / w, frame_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized_img = img.resize((new_w, new_h), Image.LANCZOS)
            self.displayed_image = ctk.CTkImage(resized_img, size=(new_w, new_h))
            self.image_label.configure(image=self.displayed_image)

    # Processing
    def _process_image(self):
        self.original_image = None  # Free memory

        self.tile_count_label_value.configure(text=f" {len(self.tiles)}")
        self.tile_results = [None] * len(self.tiles)

        self.tile_listbox.delete(0, tk.END)
        for i in range(len(self.tiles)):
            self.tile_listbox.insert(tk.END, f"Tile {i + 1}")

        self.progress_bar.set(0)
        thread = threading.Thread(target=self._process_tiles, daemon=True)
        thread.start()

    def _process_tiles(self):
        """Run YOLO inference on every tile (background thread)."""
        if not self.tiles:
            return

        self.tile_results = [None] * len(self.tiles)
        total_objects = 0
        start_time = time.time()

        for i, tile_path in enumerate(self.tiles):
            self.status_label.configure(text=f"Processing tile {i + 1}/{len(self.tiles)}...")
            self.progress_bar.set(i / len(self.tiles))

            conf = self.model_detect_param["conf"] / 100
            iou = self.model_detect_param["iou"] / 100

            try:
                results = self.model(tile_path, conf=conf, iou=iou)
                self.tile_results[i] = results[0]
                total_objects += len(results[0].boxes)
            except Exception as e:
                print(f"Error processing tile {i + 1}: {e}")
                self.tile_results[i] = None

        total_time = time.time() - start_time
        self.after(0, lambda: self._update_after_processing(total_objects, total_time))

    def _update_after_processing(self, total_objects, total_time):
        """Update UI after all tiles have been processed."""
        self.status_label.configure(text="Processing complete")
        self.progress_bar.set(1)

        self.total_objects = total_objects
        self.inference_time = total_time

        self.total_objects_label_value.configure(text=f" {total_objects}")
        self.inference_time_label_value.configure(text=f" {total_time:.2f}s")

        self._update_production_calc()

        if self.tiles:
            self.prev_btn.configure(state=tk.NORMAL)
            self.next_btn.configure(state=tk.NORMAL)
            self.current_tile_index = 0
            self._show_current_tile()

    # Production calculation
    def _update_production_calc(self):
        area = float(self.prod_calc_var["area"])
        tbs_per_tree = float(self.prod_calc_var["tbs"])

        tbs = self.total_objects * tbs_per_tree
        self.production = round(tbs / area) if area else 0

        self.result_area_label.configure(text=f"Plantation Area : {area} ha")
        self.result_production_label.configure(
            text=f"Estimated Plantation Oil Palm Production : {self.production} ton/ha/tahun",
        )

    # Tile navigation

    def _show_current_tile(self):
        """Display the current tile with its detections."""
        if not self.tiles or self.current_tile_index >= len(self.tiles):
            return

        img_path = self.tiles[self.current_tile_index]

        if self.tile_results and self.tile_results[self.current_tile_index] is not None:
            tile = cv2.imread(img_path)
            display_img = tile.copy()
            result = self.tile_results[self.current_tile_index]
            display_img = draw_detections(display_img, result)

            object_count = len(result.boxes)
            self.objects_count_label_value.configure(text=f" {object_count}")
            self._display_image(display_img)
        else:
            self.objects_count_label_value.configure(text=" 0")
            self._display_image(img_path)

        self.current_tile_label_value.configure(
            text=f" {self.current_tile_index + 1}/{len(self.tiles)}",
        )

        self.tile_listbox.selection_clear(0, tk.END)
        self.tile_listbox.selection_set(self.current_tile_index)
        self.tile_listbox.see(self.current_tile_index)

        self.prev_btn.configure(
            state=tk.NORMAL if self.current_tile_index > 0 else tk.DISABLED)
        self.next_btn.configure(
            state=tk.NORMAL if self.current_tile_index < len(self.tiles) - 1 else tk.DISABLED)

    def _show_previous_tile(self):
        if self.current_tile_index > 0:
            self.current_tile_index -= 1
            self._show_current_tile()

    def _show_next_tile(self):
        if self.current_tile_index < len(self.tiles) - 1:
            self.current_tile_index += 1
            self._show_current_tile()

    def _on_tile_selected(self, event):
        selection = self.tile_listbox.curselection()
        if selection:
            self.current_tile_index = selection[0]
            self._show_current_tile()

    # File operations

    def _save_project(self):
        save_project_csv(
            tile_results=self.tile_results,
            project_name=self.project_name,
            model_name=self.model_custom,
            image_name=self.image_name,
            total_tiles=len(self.tiles),
            total_objects=self.total_objects,
            inference_time=self.inference_time,
            plantation_area=self.prod_calc_var["area"],
            production=self.production,
        )

    def _save_image_detects(self):
        save_detection_images(self.tiles, self.tile_results, self.project_name)

    def _stitch_tiles(self):
        stitch_tiles(self.tiles, self.tile_results, self.project_name, draw_detections)

    def _new_project(self):
        if os.path.exists("split_images"):
            shutil.rmtree("split_images")
        self._init_state()

    def _on_exit(self):
        try:
            if os.path.exists("split_images"):
                shutil.rmtree("split_images")
        except Exception as e:
            print(f"Error deleting directory: {e}")
        finally:
            self.destroy()


def on_exit():
    try:
        if os.path.exists("split_images"):
            shutil.rmtree("split_images")
    except Exception as e:
        print(f"Error deleting directory: {e}")
    finally:
        app.destroy()


if __name__ == "__main__":
    app = YOLOv8TiledApp()
    app.protocol("WM_DELETE_WINDOW", on_exit)
    app.mainloop()