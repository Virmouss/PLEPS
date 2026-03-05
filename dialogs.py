import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO


class InitWindow(ctk.CTkToplevel):
    """Modal dialog for entering the project name on startup."""

    def __init__(self, master=None, title="New Project", text="Enter value:"):
        super().__init__(master)
        self.title(title)
        self.geometry("300x150")
        self.lift()
        self.attributes("-topmost", True)
        self.grab_set()

        self.result = None

        self.label = ctk.CTkLabel(self, text=text)
        self.label.pack(padx=20, pady=10)

        self.entry = ctk.CTkEntry(self)
        self.entry.pack(padx=20, pady=5, fill="x")
        self.entry.focus()

        self.ok_button = ctk.CTkButton(self, text="OK", command=self._ok_event)
        self.ok_button.pack(padx=20, pady=10)

    def _ok_event(self):
        self.result = self.entry.get()
        self.destroy()

    def get_input(self):
        self.master.wait_window(self)
        return self.result


class ModelSelectWindow(ctk.CTkToplevel):
    """Dialog for selecting a built-in or custom YOLO model."""

    def __init__(self, master, selected_option):
        super().__init__(master)
        self.title("Model Select")
        self.geometry("400x350")

        self.selected_option = selected_option
        self.radio_var = ctk.StringVar(value=self.selected_option["value"])
        self.custom_model = {"value": "", "text": ""}

        self.options = [
            {"text": "YoloV8n", "value": "models/yolov8n.pt"},
            {"text": "YoloV8s", "value": "models/yolov8s.pt"},
            {"text": "YoloV8m", "value": "models/yolov8m.pt"},
        ]

        # --- Options frame ---
        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(pady=20, padx=20, fill="both", expand=True)

        ctk.CTkLabel(self.options_frame, text="Select an option:").pack(pady=10)

        for option in self.options:
            rb = ctk.CTkRadioButton(
                self.options_frame,
                text=option["text"],
                variable=self.radio_var,
                value=option["value"],
            )
            rb.pack(pady=5, anchor="w", padx=20)

        # --- Custom model row ---
        self.custom_model_frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        self.custom_model_frame.pack(pady=10, fill="x")

        self.custom_btn = ctk.CTkButton(
            self.custom_model_frame,
            text="Load Custom Model",
            command=self._load_custom_model,
        )
        self.custom_btn.pack(side="left", padx=10, pady=10)

        self.custom_status_label = ctk.CTkLabel(
            self.custom_model_frame, text="", text_color="green"
        )
        self.custom_status_label.pack(side="left")

        # --- Apply / Cancel ---
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)

        self.apply_btn = ctk.CTkButton(
            self.button_frame, text="Apply", command=self._apply_selection
        )
        self.apply_btn.pack(side="left", padx=10)

        self.cancel_btn = ctk.CTkButton(
            self.button_frame, text="Cancel", command=self.destroy
        )
        self.cancel_btn.pack(side="left", padx=10)

    def _load_custom_model(self):
        model_path = filedialog.askopenfilename(
            title="Select YOLOv8 Model",
            filetypes=[("YOLOv8 Models", "*.pt *.pth *.onnx")],
        )
        if model_path:
            try:
                model_base = os.path.basename(model_path)
                self.custom_status_label.configure(
                    text="Custom Model Loaded", text_color="green"
                )
                self.custom_model["text"] = model_base
                self.custom_model["value"] = model_path
            except Exception as e:
                print(e)
                self.custom_model["text"] = ""
                self.custom_model["value"] = ""

    def _apply_selection(self):
        if self.custom_model["value"] != "":
            model_path = self.custom_model["value"]
            model_name = self.custom_model["text"]

            self.master.model = YOLO(model_path)
            self.master.model_status_label_value.configure(text=f"{model_name}")
            self.master.model_custom = model_name
            self.destroy()
        else:
            selected_value = self.radio_var.get()
            selected_text = next(
                (opt["text"] for opt in self.options if opt["value"] == selected_value),
                "No option selected",
            )

            self.selected_option["value"] = selected_value
            self.selected_option["text"] = selected_text

            self.master.model_status_label_value.configure(text=f"{selected_text}")
            self.master.model_custom = selected_text
            self.master.model = YOLO(selected_value)
            self.destroy()


class DetectParamWindow(ctk.CTkToplevel):

    def __init__(self, master, params):
        super().__init__(master)
        self.title("Edit Detection Parameters")
        self.geometry("400x300")

        self.params = params

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Confidence
        ctk.CTkLabel(self, text="Confidence Threshold:").grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="w"
        )
        self.conf_entry = ctk.CTkEntry(self)
        self.conf_entry.grid(row=0, column=1, padx=(0, 20), pady=(20, 10), sticky="ew")
        self.conf_entry.insert(0, str(self.params["conf"]))

        # IOU
        ctk.CTkLabel(self, text="IOU Threshold:").grid(
            row=1, column=0, padx=20, pady=10, sticky="w"
        )
        self.iou_entry = ctk.CTkEntry(self)
        self.iou_entry.grid(row=1, column=1, padx=(0, 20), pady=10, sticky="ew")
        self.iou_entry.insert(0, str(self.params["iou"]))

        # Validation
        self.validation_label = ctk.CTkLabel(self, text="", text_color="red")
        self.validation_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # Buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=20, sticky="sew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)

        self.apply_btn = ctk.CTkButton(
            self.button_frame, text="Apply", command=self._apply_changes
        )
        self.apply_btn.grid(row=0, column=0, padx=10, sticky="ew")

        self.cancel_btn = ctk.CTkButton(
            self.button_frame, text="Cancel", command=self.destroy
        )
        self.cancel_btn.grid(row=0, column=1, padx=10, sticky="ew")

    def _validate_inputs(self):
        try:
            conf = int(self.conf_entry.get())
            iou = int(self.iou_entry.get())

            if not (0 <= conf <= 100):
                raise ValueError("Confidence must be between 0-100")
            if not (0 <= iou <= 100):
                raise ValueError("IOU must be between 0-100")

            return True, conf, iou
        except ValueError as e:
            self.validation_label.configure(text=str(e))
            return False, None, None

    def _apply_changes(self):
        valid, conf, iou = self._validate_inputs()
        if valid:
            self.params["conf"] = conf
            self.params["iou"] = iou
            self.master.model_detect_param["conf"] = conf
            self.master.model_detect_param["iou"] = iou
            self.destroy()


class CalcSettingWindow(ctk.CTkToplevel):
    def __init__(self, master, var):
        super().__init__(master)
        self.title("Production Calculation")
        self.geometry("400x300")

        self.var = var

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Area
        ctk.CTkLabel(self, text="Area (ha):").grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="w"
        )
        self.area_entry = ctk.CTkEntry(self)
        self.area_entry.grid(row=0, column=1, padx=(0, 20), pady=(20, 10), sticky="ew")
        self.area_entry.insert(0, str(self.var["area"]))

        # TBS
        ctk.CTkLabel(self, text="TBS Satu Pohon (ton/Tahun):").grid(
            row=1, column=0, padx=20, pady=10, sticky="w"
        )
        self.tbs_entry = ctk.CTkEntry(self)
        self.tbs_entry.grid(row=1, column=1, padx=(0, 20), pady=10, sticky="ew")
        self.tbs_entry.insert(0, str(self.var["tbs"]))

        # Validation
        self.validation_label = ctk.CTkLabel(self, text="", text_color="red")
        self.validation_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # Buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=20, sticky="sew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)

        self.apply_btn = ctk.CTkButton(
            self.button_frame, text="Apply", command=self._apply_changes
        )
        self.apply_btn.grid(row=0, column=0, padx=10, sticky="ew")

        self.cancel_btn = ctk.CTkButton(
            self.button_frame, text="Cancel", command=self.destroy
        )
        self.cancel_btn.grid(row=0, column=1, padx=10, sticky="ew")

    def _validate_inputs(self):
        try:
            area = float(self.area_entry.get())
            tbs = float(self.tbs_entry.get())
            return True, area, tbs
        except ValueError as e:
            self.validation_label.configure(text=str(e))
            return False, None, None

    def _apply_changes(self):
        valid, area, tbs = self._validate_inputs()
        if valid:
            self.var["area"] = area
            self.var["tbs"] = tbs
            self.master.prod_calc_var["area"] = area
            self.master.prod_calc_var["tbs"] = tbs
            self.destroy()


class AboutWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("About")
        self.geometry("400x300")
        self.resizable(False, False)

        self.body_frame = ctk.CTkFrame(self)
        self.body_frame.pack(pady=5, padx=20, fill="both", expand=True)

        ctk.CTkLabel(
            self.body_frame,
            text="Perangkat Lunak Estimasi Produksi Sawit",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10, padx=10)

        textbox = ctk.CTkTextbox(
            self.body_frame, width=400, height=50, fg_color="#2b2b2b", wrap="word"
        )
        textbox.pack(pady=20, padx=10)
        textbox.tag_config("center", justify="center")

        text_content = (
            "Aplikasi ini bertujuan untuk mengestimasi produktivitas "
            "perkebunan kelapa sawit dari citra UAV"
        )
        textbox.insert("1.0", text_content)
        textbox.tag_add("center", "1.0", tk.END)
        textbox.configure(state="disabled")

        ctk.CTkLabel(
            self.body_frame, text="Ver 1.0", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=1, padx=10)

        ctk.CTkLabel(
            self.body_frame,
            text="Created By Izzan Alfadhil",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=5, padx=10)

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)

        self.apply_btn = ctk.CTkButton(
            self.button_frame, text="Close", command=self.destroy
        )
        self.apply_btn.pack(side="left", padx=10)


class UserManualWindow(ctk.CTkToplevel):

    _MANUAL_TEXT = """\
Langkah 1 : Memuat dan Memproses Gambar

    1. Klik tombol "Load Image" pada panel kiri.

    2. Pilih file citra perkebunan Anda (format .jpg, .png, .tif, dll.).

    3. Gambar akan ditampilkan di area "Detection Results". Jika gambar berukuran besar, sistem akan secara otomatis membaginya menjadi beberapa bagian (tile).

    4. Setelah gambar dimuat, klik tombol "Process Image".

    5. Perhatikan progress bar di bagian kanan bawah. Proses ini mungkin memerlukan waktu beberapa saat.

    6. Setelah selesai, kotak-kotak deteksi berwarna merah akan muncul pada gambar dan semua kolom metrik akan terisi.

    Langkah 2: Konfigurasi (Opsional)

    Sebelum memproses gambar, Anda dapat menyesuaikan beberapa pengaturan untuk hasil yang lebih akurat.

    - Ubah Model Deteksi: Buka menu Setting > Model. Pilih salah satu model bawaan (YoloV8n, YoloV8s, YoloV8m) atau klik "Load Custom Model" untuk menggunakan file model .pt Anda sendiri. Klik "Apply" untuk menyimpan.

    - Ubah Parameter Deteksi: Buka menu Setting > Detection Parameter. Atur nilai "Confidence Threshold" (tingkat keyakinan) dan "IOU Threshold" (tingkat tumpang tindih). Klik "Apply" untuk menyimpan.

    - Atur Parameter Kalkulasi: Klik tombol "Calculation Settings". Masukkan data "Area (ha)" dan "TBS Satu Pohon (ton/Tahun)". Klik "Apply".

    Langkah 3: Memuat dan Memproses Gambar

    1. Klik tombol "Load Image" pada panel kiri.

    2. Pilih file citra perkebunan Anda (format .jpg, .png, .tif, dll.).

    3. Gambar akan ditampilkan di area "Detection Results". Jika gambar berukuran besar, sistem akan secara otomatis membaginya menjadi beberapa bagian (tile).

    4. Setelah gambar dimuat, klik tombol "Process Image".

    5. Perhatikan progress bar di bagian kanan bawah. Proses ini mungkin memerlukan waktu beberapa saat.

    6. Setelah selesai, kotak-kotak deteksi berwarna merah akan muncul pada gambar dan semua kolom metrik akan terisi.

    Langkah 4: Menganalisis Hasil

    1. Gunakan tombol "◀ Previous" dan "Next ▶" atau klik item pada "Tile List" untuk menavigasi dan melihat hasil deteksi pada setiap tile.

    2. Perhatikan panel "Detection Metrics" untuk melihat jumlah total objek (Total objects) dan waktu pemrosesan (Total inference time).

    3. Lihat jumlah objek pada tile yang sedang aktif pada label "Objects in tile".

    4. Periksa hasil akhir pada bagian "Estimated Plantation Oil Palm Production".

    Langkah 5: Mengekspor Hasil

    Buka menu File untuk menyimpan hasil analisis Anda:

    - Save: Untuk menyimpan laporan data proyek lengkap dalam format file .csv.

    - Save Image Detections: Untuk menyimpan setiap tile yang sudah diberi kotak deteksi sebagai file gambar terpisah.

    - Stitch: Untuk menggabungkan semua tile hasil deteksi menjadi satu gambar besar yang utuh.

    Langkah 6: Proyek Baru atau Keluar

    - Untuk memulai analisis baru, pilih File > New.

    - Untuk keluar dari aplikasi, pilih File > Exit atau klik tombol 'X' pada jendela.

        """

    def __init__(self, master):
        super().__init__(master)
        self.title("User Manual")
        self.geometry("900x600")
        self.resizable(False, False)

        self.body_frame = ctk.CTkFrame(self)
        self.body_frame.pack(pady=5, padx=20, fill="both", expand=True)

        ctk.CTkLabel(
            self.body_frame,
            text="User Manual",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10, padx=10)

        textbox = ctk.CTkTextbox(
            self.body_frame, width=900, height=400, fg_color="#2b2b2b", wrap="word"
        )
        textbox.pack(pady=20, padx=10)
        textbox.tag_config("left", justify="left")
        textbox.insert("1.0", self._MANUAL_TEXT)
        textbox.tag_add("left", "1.0", tk.END)
        textbox.configure(state="disabled")

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)

        self.apply_btn = ctk.CTkButton(
            self.button_frame, text="Close", command=self.destroy
        )
        self.apply_btn.pack(side="left", padx=10)
