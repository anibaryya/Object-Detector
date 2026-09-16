import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import time
import torch
from ultralytics import YOLO
from collections import Counter

# Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COCO_CLASS_COUNT = 80
WEBCAM_MODEL = "yolov8s.pt"
IMAGE_MODEL = "yolov8x.pt"
DAILY_OBJECTS_WEIGHTS = Path("runs/detect/daily_objects/weights/best.pt")
WEBCAM_IMGSZ = 480
IMAGE_IMGSZ = 960
WEBCAM_CONF = 0.35
IMAGE_CONF = 0.5
IOU = 0.45
MAX_DET = 300
DISPLAY_INTERVAL = 0.033  # ~30 UI updates/sec


def configure_gpu():
    """Enable PyTorch/CUDA optimizations when a GPU is available."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return "cuda"
    return "cpu"


def resolve_model_path(model_type):
    """Return weight path for the selected model type."""
    if model_type == "daily":
        if DAILY_OBJECTS_WEIGHTS.is_file():
            return str(DAILY_OBJECTS_WEIGHTS)
        return WEBCAM_MODEL
    return WEBCAM_MODEL


class ObjectDetectorUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VisionAI — Surveillance & Inventory Detection")
        self.geometry("1280x820")
        self.resizable(True, True)

        self.model = None
        self.model_type = "coco"
        self.device = configure_gpu()
        self.use_half = self.device == "cuda"
        self.is_detecting = False
        self.cap = None
        self.image_path = None

        self.min_confidence = WEBCAM_CONF
        self._ui_update_scheduled = False
        self._pending_display = None

        # Webcam pipeline state (capture + inference threads)
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._latest_frame = None
        self._display_frame = None
        self._latest_fps = 0.0
        self._latest_objects = []
        self._model_loading = False

        self.setup_ui()
        self.load_model_async("coco")

    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self, corner_radius=0)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel — controls
        self.left_panel = ctk.CTkFrame(self.main_container, width=320, corner_radius=15)
        self.left_panel.pack(side="left", fill="y", padx=(0, 10))
        self.left_panel.pack_propagate(False)

        title_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        title_frame.pack(pady=(20, 10), padx=20, fill="x")

        ctk.CTkLabel(
            title_frame,
            text="VisionAI Detector",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack()

        ctk.CTkLabel(
            title_frame,
            text="Surveillance & Inventory Management",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(pady=(4, 0))

        self.subtitle_label = ctk.CTkLabel(
            title_frame,
            text=f"YOLOv8s live • {COCO_CLASS_COUNT} COCO classes • PyTorch",
            font=ctk.CTkFont(size=11),
            text_color="#5dade2",
        )
        self.subtitle_label.pack(pady=(6, 0))

        # Model status
        self.status_frame = ctk.CTkFrame(self.left_panel, corner_radius=10)
        self.status_frame.pack(pady=10, padx=20, fill="x")

        self.status_indicator = ctk.CTkLabel(
            self.status_frame, text="⚡", font=ctk.CTkFont(size=20)
        )
        self.status_indicator.pack(side="left", padx=10, pady=10)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Loading model...",
            font=ctk.CTkFont(size=12),
        )
        self.status_label.pack(side="left", pady=10)

        # Model selector
        model_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        model_frame.pack(pady=(5, 5), padx=20, fill="x")

        ctk.CTkLabel(
            model_frame,
            text="Detection Model",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(0, 8))

        self.model_var = tk.StringVar(value="coco")
        self.model_hint = ctk.CTkLabel(
            model_frame,
            text="COCO pretrained — 80 general object classes",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=260,
            justify="left",
        )
        self.model_hint.pack(pady=(0, 8))

        ctk.CTkRadioButton(
            model_frame,
            text="🌐 Pretrained COCO",
            variable=self.model_var,
            value="coco",
            font=ctk.CTkFont(size=14),
            command=self.on_model_type_changed,
        ).pack(anchor="w", pady=3)

        ctk.CTkRadioButton(
            model_frame,
            text="🏠 Daily Objects (custom)",
            variable=self.model_var,
            value="daily",
            font=ctk.CTkFont(size=14),
            command=self.on_model_type_changed,
        ).pack(anchor="w", pady=3)

        # Application use-case mode
        use_case_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        use_case_frame.pack(pady=(10, 5), padx=20, fill="x")

        ctk.CTkLabel(
            use_case_frame,
            text="Application Mode",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(0, 8))

        self.use_case_var = tk.StringVar(value="surveillance")
        self.use_case_hint = ctk.CTkLabel(
            use_case_frame,
            text="Live monitoring — track people, vehicles, and objects",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=260,
            justify="left",
        )
        self.use_case_hint.pack(pady=(0, 8))

        ctk.CTkRadioButton(
            use_case_frame,
            text="🛡️ Surveillance",
            variable=self.use_case_var,
            value="surveillance",
            font=ctk.CTkFont(size=14),
            command=self.update_use_case_hint,
        ).pack(anchor="w", pady=3)

        ctk.CTkRadioButton(
            use_case_frame,
            text="📦 Inventory Management",
            variable=self.use_case_var,
            value="inventory",
            font=ctk.CTkFont(size=14),
            command=self.update_use_case_hint,
        ).pack(anchor="w", pady=3)

        # Input mode — webcam vs static image
        mode_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        mode_frame.pack(pady=15, padx=20, fill="x")

        ctk.CTkLabel(
            mode_frame,
            text="Input Mode",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(0, 8))

        self.mode_var = tk.StringVar(value="webcam")

        ctk.CTkRadioButton(
            mode_frame,
            text="📹 Live Webcam",
            variable=self.mode_var,
            value="webcam",
            font=ctk.CTkFont(size=14),
            command=self.switch_input_mode,
        ).pack(anchor="w", pady=3)

        ctk.CTkRadioButton(
            mode_frame,
            text="🖼️ Static Image",
            variable=self.mode_var,
            value="image",
            font=ctk.CTkFont(size=14),
            command=self.switch_input_mode,
        ).pack(anchor="w", pady=3)

        # Actions
        action_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        action_frame.pack(pady=10, padx=20, fill="x")

        self.start_btn = ctk.CTkButton(
            action_frame,
            text="▶ Start Detection",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            corner_radius=10,
            command=self.toggle_detection,
            state="disabled",
        )
        self.start_btn.pack(fill="x", pady=5)

        self.browse_btn = ctk.CTkButton(
            action_frame,
            text="📁 Browse Image",
            font=ctk.CTkFont(size=16),
            height=45,
            corner_radius=10,
            fg_color="gray30",
            hover_color="gray25",
            command=self.browse_image,
            state="disabled",
        )
        self.browse_btn.pack(fill="x", pady=5)

        # Confidence threshold
        settings_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        settings_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(
            settings_frame,
            text="⚙ Detection Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(0, 8))

        self.conf_label = ctk.CTkLabel(
            settings_frame,
            text=f"Confidence threshold: {self.min_confidence:.2f}",
            font=ctk.CTkFont(size=12),
        )
        self.conf_label.pack()

        self.conf_slider = ctk.CTkSlider(
            settings_frame,
            from_=0.1,
            to=0.95,
            number_of_steps=17,
            command=self.update_confidence,
        )
        self.conf_slider.set(self.min_confidence)
        self.conf_slider.pack(fill="x", pady=5)

        # Stats panel
        self.stats_frame = ctk.CTkFrame(self.left_panel, corner_radius=10)
        self.stats_frame.pack(pady=(10, 20), padx=20, fill="both", expand=True)

        self.stats_title = ctk.CTkLabel(
            self.stats_frame,
            text="📊 Surveillance Feed",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.stats_title.pack(pady=10)

        self.stats_text = ctk.CTkTextbox(
            self.stats_frame,
            height=160,
            font=ctk.CTkFont(size=12),
            corner_radius=10,
        )
        self.stats_text.pack(padx=10, pady=(0, 10), fill="both", expand=True)

        # Right panel — video/image display
        self.right_panel = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.right_panel.pack(side="right", fill="both", expand=True)

        display_header = ctk.CTkFrame(self.right_panel, height=50, corner_radius=0)
        display_header.pack(fill="x", padx=20, pady=(20, 10))

        self.fps_label = ctk.CTkLabel(
            display_header,
            text="FPS: --",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00ff00",
        )
        self.fps_label.pack(side="left")

        self.class_count_label = ctk.CTkLabel(
            display_header,
            text=f"{COCO_CLASS_COUNT} object classes",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        self.class_count_label.pack(side="left", padx=(20, 0))

        gpu_text = f"PyTorch • {self.device.upper()}"
        if self.use_half:
            gpu_text += " • FP16"
        self.device_label = ctk.CTkLabel(
            display_header,
            text=gpu_text,
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        self.device_label.pack(side="right")

        self.display_frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        self.display_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.video_label = ctk.CTkLabel(
            self.display_frame,
            text=(
                "Video feed will appear here\n\n"
                "Choose Surveillance or Inventory mode,\n"
                "select Live Webcam or Static Image, then start detection"
            ),
            font=ctk.CTkFont(size=16),
            text_color="gray",
        )
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

    def on_model_type_changed(self):
        model_type = self.model_var.get()
        if model_type == "daily":
            if DAILY_OBJECTS_WEIGHTS.is_file():
                self.model_hint.configure(
                    text="Custom daily-use objects (pen, toothbrush, keys, …)"
                )
            else:
                self.model_hint.configure(
                    text="Custom weights not found — train with train_model.py"
                )
        else:
            self.model_hint.configure(text="COCO pretrained — 80 general object classes")

        if self.is_detecting:
            messagebox.showinfo(
                "Model Change",
                "Stop detection first, then switch models.",
            )
            self.model_var.set(self.model_type)
            return

        self.load_model_async(model_type)

    def update_use_case_hint(self):
        if self.use_case_var.get() == "inventory":
            self.use_case_hint.configure(
                text="Stock counting — classify and tally detected items"
            )
            self.stats_title.configure(text="📦 Inventory Count")
        else:
            self.use_case_hint.configure(
                text="Live monitoring — track people, vehicles, and objects"
            )
            self.stats_title.configure(text="📊 Surveillance Feed")

    def load_model_async(self, model_type):
        if self._model_loading:
            return
        self._model_loading = True
        self.model_type = model_type
        self.start_btn.configure(state="disabled")
        self.status_indicator.configure(text="⚡")
        self.status_label.configure(text="Loading model...")

        def load():
            try:
                weight_path = resolve_model_path(model_type)
                model = YOLO(weight_path)
                if self.device == "cuda":
                    model.to("cuda")
                self.model = model
                class_count = len(model.names)
                using_custom = (
                    model_type == "daily" and DAILY_OBJECTS_WEIGHTS.is_file()
                )
                self.after(
                    0,
                    lambda: self._on_model_loaded(class_count, model_type, using_custom),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Failed to load model:\n{exc}"
                    ),
                )
                self.after(0, self._on_model_failed)
            finally:
                self._model_loading = False

        threading.Thread(target=load, daemon=True).start()

    def _on_model_loaded(self, class_count, model_type, using_custom):
        if using_custom:
            label = f"Daily Objects ready ({class_count} classes)"
            subtitle = f"Custom daily objects • {class_count} classes • PyTorch"
        elif model_type == "daily":
            label = f"COCO fallback ({class_count} classes) — train custom model"
            subtitle = f"YOLOv8s fallback • {class_count} COCO classes • PyTorch"
        else:
            label = f"YOLOv8s ready ({class_count} classes)"
            subtitle = f"YOLOv8s live / YOLOv8x images • {class_count} classes • PyTorch"

        self.status_label.configure(text=label)
        self.status_indicator.configure(text="✅")
        self.subtitle_label.configure(text=subtitle)
        self.class_count_label.configure(text=f"{class_count} object classes")
        self.start_btn.configure(state="normal")

    def _on_model_failed(self):
        self.status_label.configure(text="Model failed to load")
        self.status_indicator.configure(text="❌")

    def switch_input_mode(self):
        mode = self.mode_var.get()
        if mode == "image":
            self.browse_btn.configure(state="normal")
            self.start_btn.configure(text="🔍 Detect Objects")
            self.min_confidence = IMAGE_CONF
        else:
            self.browse_btn.configure(state="disabled")
            self.start_btn.configure(text="▶ Start Detection")
            self.min_confidence = WEBCAM_CONF
        self.conf_slider.set(self.min_confidence)
        self.conf_label.configure(
            text=f"Confidence threshold: {self.min_confidence:.2f}"
        )

    def update_confidence(self, value):
        self.min_confidence = float(value)
        self.conf_label.configure(
            text=f"Confidence threshold: {self.min_confidence:.2f}"
        )

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.image_path = file_path
            self.display_static_preview(file_path)

    def display_static_preview(self, path):
        image = Image.open(path)
        self._show_pil_image(image)

    def _show_pil_image(self, image):
        display_size = (
            max(self.display_frame.winfo_width() - 20, 400),
            max(self.display_frame.winfo_height() - 20, 300),
        )
        image = image.copy()
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def toggle_detection(self):
        if self.model is None:
            messagebox.showwarning("Model Loading", "Please wait for the model to load.")
            return

        if self.mode_var.get() == "webcam":
            if not self.is_detecting:
                self.start_webcam_detection()
            else:
                self.stop_webcam_detection()
        elif self.image_path:
            self.detect_image()
        else:
            messagebox.showwarning("No Image", "Please select an image first.")

    def start_webcam_detection(self):
        self.is_detecting = True
        self._latest_frame = None
        self._display_frame = None
        self._latest_fps = 0.0
        self._latest_objects = []
        self.start_btn.configure(text="⏹ Stop Detection")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open webcam.")
            self.is_detecting = False
            self.start_btn.configure(text="▶ Start Detection")
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        threading.Thread(target=self._display_loop, daemon=True).start()

    def stop_webcam_detection(self):
        self.is_detecting = False
        self.start_btn.configure(text="▶ Start Detection")

        def cleanup():
            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_label.configure(image="", text="Detection stopped")
            self.fps_label.configure(text="FPS: --")

        self.after(100, cleanup)

    def _capture_loop(self):
        """Continuously grab the latest frame; never block on inference."""
        while self.is_detecting and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)

    def _inference_loop(self):
        """Run YOLO on the most recent frame only; skip stale frames."""
        fps_history = []

        while self.is_detecting:
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            infer_frame = frame.copy()
            start_time = time.perf_counter()
            results = self.predict(infer_frame, WEBCAM_IMGSZ)
            detected_objects = self.draw_detections(infer_frame, results)

            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_history.append(fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            with self._result_lock:
                self._display_frame = infer_frame
                self._latest_fps = avg_fps
                self._latest_objects = detected_objects

    def _display_loop(self):
        """Push annotated frames to the UI at a capped rate."""
        while self.is_detecting:
            with self._result_lock:
                frame = self._display_frame
                fps = self._latest_fps
                objects = list(self._latest_objects)

            if frame is not None:
                self.schedule_display_update(frame, fps, objects)

            time.sleep(DISPLAY_INTERVAL)

    def predict(self, frame, imgsz):
        kwargs = {
            "conf": self.min_confidence,
            "iou": IOU,
            "device": self.device,
            "imgsz": imgsz,
            "verbose": False,
            "max_det": MAX_DET,
            "agnostic_nms": False,
        }
        if self.use_half:
            kwargs["half"] = True
        return self.model.predict(frame, **kwargs)

    def draw_detections(self, frame, results, thickness=2):
        detected_objects = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                confidence = float(box.conf)
                label = self.model.names[class_id]
                detected_objects.append(label)

                color = (0, 255, 0)
                if self.use_case_var.get() == "surveillance" and label == "person":
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label_text = f"{label} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    frame, (x1, y1 - text_height - 8), (x1 + text_width + 4, y1), color, -1
                )
                cv2.putText(
                    frame,
                    label_text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                )
        return detected_objects

    def detect_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load the selected image.")
            return

        start_time = time.perf_counter()

        # Use heavier model for static images when on COCO preset
        if self.model_type == "coco":
            try:
                image_model = YOLO(IMAGE_MODEL)
                if self.device == "cuda":
                    image_model.to("cuda")
                saved_model = self.model
                self.model = image_model
                results = self.predict(image, IMAGE_IMGSZ)
                self.model = saved_model
            except Exception:
                results = self.predict(image, IMAGE_IMGSZ)
        else:
            results = self.predict(image, IMAGE_IMGSZ)

        detected_objects = self.draw_detections(image, results, thickness=3)
        elapsed = time.perf_counter() - start_time

        self.schedule_display_update(
            image, 1.0 / elapsed if elapsed > 0 else 0, detected_objects
        )

        output_path = "detected_result.jpg"
        cv2.imwrite(output_path, image)
        messagebox.showinfo(
            "Detection Complete",
            f"Found {len(detected_objects)} object(s).\nSaved to: {output_path}",
        )

    def schedule_display_update(self, frame, fps, detected_objects):
        """Coalesce UI updates — always show the latest frame, drop stale ones."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        stats_text = self.build_stats_text(detected_objects)
        self._pending_display = (image, fps, stats_text)

        if self._ui_update_scheduled:
            return

        self._ui_update_scheduled = True
        self.after(0, self._flush_display_update)

    def _flush_display_update(self):
        self._ui_update_scheduled = False
        if self._pending_display is None:
            return

        image, fps, stats_text = self._pending_display
        self._show_pil_image(image)

        if fps > 0:
            fps_int = int(round(fps))
            color = "#00ff00" if fps_int >= 60 else "#ffcc00" if fps_int >= 30 else "#ff6666"
            self.fps_label.configure(text=f"FPS: {fps_int}", text_color=color)

        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", stats_text)

    def build_stats_text(self, detected_objects):
        if not detected_objects:
            return "No objects detected above confidence threshold."

        counts = Counter(detected_objects)
        use_case = self.use_case_var.get()

        if use_case == "inventory":
            header = "Inventory Tally\n" + "-" * 22 + "\n"
            lines = [f"• {obj}: {count}" for obj, count in counts.most_common()]
            footer = f"\nSKU count: {len(detected_objects)} items\nUnique classes: {len(counts)}"
        else:
            header = "Surveillance Detections\n" + "-" * 22 + "\n"
            lines = [f"• {obj}: {count}" for obj, count in counts.most_common()]
            persons = counts.get("person", 0)
            footer = f"\nTotal tracked: {len(detected_objects)}"
            if persons:
                footer += f"\n⚠ Persons in frame: {persons}"

        return header + "\n".join(lines) + footer


if __name__ == "__main__":
    app = ObjectDetectorUI()
    app.mainloop()
