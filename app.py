"""
BottleVision – Liquid Level Estimation System
=============================================
Single-file PyQt5 application combining YOLO+ByteTrack bottle detection
with interactive virtual-line calibration.

Controls (when preview is focused):
    ↑ / ↓       Move target line ±1 px
    Shift+↑/↓   Move target line ±10 px
    Space        Pause / Resume
"""

import sys
import os
import time
import threading
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QPlainTextEdit, QSizePolicy, QFrame, QSplitter,
    QLineEdit, QComboBox, QSlider, QToolTip, QScrollArea,
    QStatusBar, QAction, QMenuBar, QMessageBox, QCheckBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QMutex, QTimer, QSize
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPalette, QColor, QFont, QIcon, QKeySequence
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS / DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH   = "best.pt"
DEFAULT_CONF         = 0.30
DEFAULT_IOU          = 0.45
DEFAULT_TOLERANCE    = 5
DEFAULT_GRADIENT_THR = 10
DEFAULT_GAUSS_KERNEL = 5
DEFAULT_DETECT_LINE  = 0.50   # fraction of frame width

ACCENT   = "#00C896"
BG_DARK  = "#0d0d1a"
BG_PANEL = "#12121f"
BG_CARD  = "#1a1a30"
FG_TEXT  = "#e8e8f0"
FG_DIM   = "#7070a0"
RED_COL  = "#ff4060"
YELL_COL = "#ffcc00"


# ──────────────────────────────────────────────────────────────────────────────
# LIQUID LEVEL DETECTION CORE   (from claude_final.py – unchanged logic)
# ──────────────────────────────────────────────────────────────────────────────

def detect_liquid_level(gray_img, min_gradient_threshold=10):
    if gray_img.size == 0 or gray_img.shape[0] < 2:
        return None, {}
    intensity_profile = np.mean(gray_img, axis=1)
    gradients = np.diff(intensity_profile)
    abs_gradients = np.abs(gradients)
    if len(abs_gradients) == 0:
        return None, {}
    max_gradient_idx   = np.argmax(abs_gradients)
    max_gradient_value = abs_gradients[max_gradient_idx]
    if max_gradient_value < min_gradient_threshold:
        dark_threshold = np.mean(intensity_profile) * 0.7
        liquid_region  = np.where(intensity_profile < dark_threshold)[0]
        if len(liquid_region) > 0:
            liquid_level = liquid_region[0]
        else:
            return None, {
                "intensity_profile": intensity_profile,
                "gradients": gradients,
                "max_gradient_idx": max_gradient_idx,
                "max_gradient_value": max_gradient_value,
            }
    else:
        liquid_level = max_gradient_idx + 1
    return liquid_level, {
        "intensity_profile": intensity_profile,
        "gradients": gradients,
        "max_gradient_idx": max_gradient_idx,
        "max_gradient_value": max_gradient_value,
    }


def process_roi_with_gaussian(roi_img, gauss_kernel=5, gradient_thr=10):
    if roi_img.size == 0:
        return None, None, {}
    gray    = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    k       = gauss_kernel if gauss_kernel % 2 == 1 else gauss_kernel + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    liquid_level_relative, gradient_info = detect_liquid_level(
        blurred, min_gradient_threshold=gradient_thr
    )
    return liquid_level_relative, blurred, gradient_info


# ──────────────────────────────────────────────────────────────────────────────
# BOTTLE TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class BottleTracker:
    """YOLO + ByteTrack tracker.  target_line_y is read via a callable so the
    GUI can update it in real-time without restarting the worker."""

    def __init__(self, model_path, detection_line_x_frac,
                 get_target_y, tolerance, conf, iou,
                 gauss_kernel, gradient_thr):
        from ultralytics import YOLO
        self.model              = YOLO(model_path)
        self.detection_line_x_frac = detection_line_x_frac
        self.get_target_y       = get_target_y   # callable → int
        self.tolerance          = tolerance
        self.conf               = conf
        self.iou                = iou
        self.gauss_kernel       = gauss_kernel
        self.gradient_thr       = gradient_thr

        self.processed_bottles  = set()
        self.bottle_results     = {}
        self.detection_line_x   = 0   # updated per frame
        self.stats = {
            "total_bottles": 0,
            "passed": 0,
            "rejected_low": 0,
            "not_detected": 0,
        }

    def reset(self):
        self.processed_bottles.clear()
        self.bottle_results.clear()
        self.stats = {
            "total_bottles": 0,
            "passed": 0,
            "rejected_low": 0,
            "not_detected": 0,
        }

    def has_crossed_line(self, bbox, track_id):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        return center_x >= self.detection_line_x and track_id not in self.processed_bottles

    def _classify(self, liquid_level_absolute):
        if liquid_level_absolute is None:
            return "NOT_DETECTED"
        target_y  = self.get_target_y()
        deviation = liquid_level_absolute - target_y
        return "PASS" if deviation <= self.tolerance else "REJECT_LOW"

    def process_bottle(self, frame, bbox, track_id):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        liquid_rel, processed_img, grad_info = process_roi_with_gaussian(
            roi, self.gauss_kernel, self.gradient_thr
        )
        liquid_abs = (y1 + liquid_rel) if liquid_rel is not None else None
        status     = self._classify(liquid_abs)
        result = {
            "track_id": track_id,
            "bbox": (x1, y1, x2, y2),
            "roi": roi,
            "processed_img": processed_img,
            "liquid_level_relative": liquid_rel,
            "liquid_level_absolute": liquid_abs,
            "status": status,
            "gradient_info": grad_info,
        }
        self.bottle_results[track_id] = result
        self.processed_bottles.add(track_id)
        self.stats["total_bottles"] += 1
        if status == "PASS":
            self.stats["passed"] += 1
        elif status == "REJECT_LOW":
            self.stats["rejected_low"] += 1
        else:
            self.stats["not_detected"] += 1
        return result

    def detect_and_track(self, frame):
        h, w = frame.shape[:2]
        self.detection_line_x = int(w * self.detection_line_x_frac)
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            imgsz=640,
        )
        newly_processed = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None:
                boxes     = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                confs     = r.boxes.conf.cpu().numpy()
                for bbox, tid, cf in zip(boxes, track_ids, confs):
                    if self.has_crossed_line(bbox, tid):
                        res = self.process_bottle(frame, bbox, tid)
                        newly_processed.append(res)
        return results, newly_processed

    def visualize(self, frame, yolo_results, show_ids=True):
        vis   = frame.copy()
        h, w  = vis.shape[:2]
        ty    = self.get_target_y()

        # Target line (green)
        cv2.line(vis, (0, ty), (w, ty), (0, 220, 150), 2)
        cv2.putText(vis, f"TARGET  Y={ty}", (10, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 150), 2)

        # Tolerance bands (dashed red)
        for y_tol in [ty - self.tolerance, ty + self.tolerance]:
            for i in range(0, w, 20):
                cv2.line(vis, (i, y_tol), (min(i + 10, w), y_tol), (60, 60, 220), 1)

        # Detection line (cyan vertical)
        cv2.line(vis, (self.detection_line_x, 0),
                 (self.detection_line_x, h), (220, 220, 0), 2)
        cv2.putText(vis, "DETECT", (self.detection_line_x + 5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 0), 2)

        # Detections
        if yolo_results and len(yolo_results) > 0:
            r = yolo_results[0]
            if r.boxes is not None and r.boxes.id is not None:
                boxes     = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                for bbox, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, bbox)
                    if tid in self.bottle_results:
                        rd     = self.bottle_results[tid]
                        status = rd["status"]
                        color  = ((0, 220, 80) if status == "PASS"
                                  else (0, 60, 220) if status == "REJECT_LOW"
                                  else (0, 140, 255))
                        ll = rd["liquid_level_absolute"]
                        if ll is not None:
                            cv2.line(vis, (x1, int(ll)), (x2, int(ll)), color, 2)
                    else:
                        color = (200, 200, 200)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    if show_ids:
                        lbl = f"#{tid}"
                        if tid in self.bottle_results:
                            lbl += f" {self.bottle_results[tid]['status']}"
                        cv2.putText(vis, lbl, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        # Stats overlay
        pad = 8
        sx, sy = 8, h - 68
        cv2.rectangle(vis, (sx, sy - pad), (360, h - pad), (0, 0, 0), -1)
        s = self.stats
        cv2.putText(vis,
                    f"Total:{s['total_bottles']}  Pass:{s['passed']}  "
                    f"Reject:{s['rejected_low']}  Unknown:{s['not_detected']}",
                    (sx + 4, sy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1)
        pass_rate = (s["passed"] / max(s["total_bottles"], 1)) * 100
        cv2.putText(vis, f"Pass rate: {pass_rate:.1f}%",
                    (sx + 4, sy + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 150), 1)
        return vis


# ──────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# ──────────────────────────────────────────────────────────────────────────────

class DetectionWorker(QThread):
    frame_ready    = pyqtSignal(QImage)
    log_message    = pyqtSignal(str)
    stats_updated  = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    finished_clean = pyqtSignal()

    def __init__(self, source, model_path, get_target_y,
                 conf, iou, tolerance, gauss_kernel, gradient_thr,
                 detect_line_frac, show_ids):
        super().__init__()
        self.source          = source
        self.model_path      = model_path
        self.get_target_y    = get_target_y
        self.conf            = conf
        self.iou             = iou
        self.tolerance       = tolerance
        self.gauss_kernel    = gauss_kernel
        self.gradient_thr    = gradient_thr
        self.detect_line_frac = detect_line_frac
        self.show_ids        = show_ids

        self._paused  = False
        self._running = True
        self._mutex   = QMutex()

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{ts}]  {msg}")

    def run(self):
        self._log(f"Loading model: {self.model_path}")
        try:
            tracker = BottleTracker(
                model_path          = self.model_path,
                detection_line_x_frac = self.detect_line_frac,
                get_target_y        = self.get_target_y,
                tolerance           = self.tolerance,
                conf                = self.conf,
                iou                 = self.iou,
                gauss_kernel        = self.gauss_kernel,
                gradient_thr        = self.gradient_thr,
            )
        except Exception as e:
            self.error_occurred.emit(f"Model load failed: {e}")
            return

        # Open video / camera
        try:
            src = int(self.source)
            self._log(f"Opening camera index {src}")
        except ValueError:
            src = self.source
            self._log(f"Opening video: {os.path.basename(str(src))}")

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open source: {self.source}")
            return

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps
        self._log(f"Stream opened  {int(cap.get(3))}×{int(cap.get(4))} @ {fps:.1f} fps")

        last_frame   = None
        yolo_results = None

        while self._running:
            if self._paused:
                time.sleep(0.05)
                # Still render last frame so the line updates while paused
                if last_frame is not None and yolo_results is not None:
                    vis = tracker.visualize(last_frame, yolo_results, self.show_ids)
                    self.frame_ready.emit(self._to_qimage(vis))
                continue

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.processed_bottles.clear()
                self._log("↩  Video loop restarted  (tracking IDs reset)")
                continue

            last_frame = frame.copy()
            try:
                yolo_results, newly = tracker.detect_and_track(frame)
            except Exception as e:
                self._log(f"⚠  Detection error: {e}")
                continue

            for res in newly:
                tid = res["track_id"]
                ll  = res["liquid_level_absolute"]
                st  = res["status"]
                dev = (ll - self.get_target_y()) if ll is not None else None
                dev_str = f"  dev={dev:+d}px" if dev is not None else ""
                icon = "✅" if st == "PASS" else ("❌" if st == "REJECT_LOW" else "⚠")
                self._log(f"{icon} Bottle #{tid}: {st}  "
                           f"(Y={ll if ll else 'N/A'}{dev_str})")
                self.stats_updated.emit(dict(tracker.stats))

            vis = tracker.visualize(frame, yolo_results, self.show_ids)
            self.frame_ready.emit(self._to_qimage(vis))

            elapsed = time.time() - t0
            wait    = max(0.001, delay - elapsed)
            time.sleep(wait)

        cap.release()
        self._log("Stream closed.")
        self.finished_clean.emit()

    @staticmethod
    def _to_qimage(bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# ──────────────────────────────────────────────────────────────────────────────
# PREVIEW LABEL  (scales image while keeping aspect ratio)
# ──────────────────────────────────────────────────────────────────────────────

class PreviewLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {BG_DARK}; border-radius: 6px;")
        self._placeholder()

    def _placeholder(self):
        self.setText("No source selected.\nUse the controls panel to load a video or camera.")
        self.setStyleSheet(
            f"background: {BG_DARK}; color: {FG_DIM}; font-size: 14px;"
            " border-radius: 6px; border: 1px solid #2a2a4a;"
        )

    def set_frame(self, qimage: QImage):
        pix = QPixmap.fromImage(qimage)
        self.setPixmap(
            pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


# ──────────────────────────────────────────────────────────────────────────────
# STYLED HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _btn(text, color=ACCENT, text_color="#000", min_w=80):
    b = QPushButton(text)
    b.setMinimumWidth(min_w)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet(f"""
        QPushButton {{
            background: {color};
            color: {text_color};
            font-weight: 700;
            font-size: 12px;
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
        }}
        QPushButton:hover   {{ background: {color}cc; }}
        QPushButton:pressed {{ background: {color}88; }}
        QPushButton:disabled {{ background: #2a2a4a; color: {FG_DIM}; }}
    """)
    return b


def _group(title):
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{
            font-weight: 700;
            font-size: 12px;
            color: {ACCENT};
            border: 1px solid #2a2a4a;
            border-radius: 8px;
            margin-top: 12px;
            padding: 18px 8px 8px 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            padding: 0 6px;
        }}
    """)
    return g


def _spin(min_v, max_v, val, suffix=""):
    s = QSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setStyleSheet(f"""
        QSpinBox {{
            background: {BG_DARK};
            color: {FG_TEXT};
            border: 1px solid #2a2a4a;
            border-radius: 5px;
            padding: 3px 6px;
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            width: 18px; background: #2a2a4a; border: none;
        }}
    """)
    return s


def _dspin(min_v, max_v, val, dec=2, step=0.05, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(val)
    s.setDecimals(dec)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setStyleSheet(f"""
        QDoubleSpinBox {{
            background: {BG_DARK};
            color: {FG_TEXT};
            border: 1px solid #2a2a4a;
            border-radius: 5px;
            padding: 3px 6px;
        }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 18px; background: #2a2a4a; border: none;
        }}
    """)
    return s


def _label(text, dim=False, bold=False):
    l = QLabel(text)
    color = FG_DIM if dim else FG_TEXT
    weight = "700" if bold else "400"
    l.setStyleSheet(f"color: {color}; font-weight: {weight}; font-size: 12px;")
    return l


# ──────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BottleVision  —  Liquid Level Estimation")
        self.resize(1400, 820)

        self._worker: DetectionWorker | None = None
        self._target_y       = 400          # current virtual line Y position
        self._target_y_lock  = threading.Lock()
        self._last_stats     = {}

        self._apply_theme()
        self._build_ui()
        self._connect_signals()

    # ── theme ────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        QApplication.setStyle("Fusion")
        pal = QPalette()
        for role, hex_c in [
            (QPalette.Window,        BG_DARK),
            (QPalette.WindowText,    FG_TEXT),
            (QPalette.Base,          BG_PANEL),
            (QPalette.AlternateBase, BG_CARD),
            (QPalette.Text,          FG_TEXT),
            (QPalette.Button,        BG_CARD),
            (QPalette.ButtonText,    FG_TEXT),
            (QPalette.Highlight,     ACCENT),
            (QPalette.HighlightedText, "#000"),
            (QPalette.ToolTipBase,   BG_CARD),
            (QPalette.ToolTipText,   FG_TEXT),
        ]:
            pal.setColor(role, QColor(hex_c))
        QApplication.instance().setPalette(pal)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG_DARK}; color: {FG_TEXT}; }}
            QSplitter::handle {{ background: #2a2a4a; }}
            QScrollBar:vertical {{
                background: {BG_PANEL}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #3a3a5a; border-radius: 4px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left control panel (scrollable) ─────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(340)
        left_scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG_PANEL}; border: none; border-radius: 10px; }}
            QScrollArea > QWidget > QWidget {{ background: {BG_PANEL}; }}
        """)
        left_panel = QWidget()
        left_panel.setStyleSheet(f"background: {BG_PANEL}; border-radius: 10px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(6)

        # Logo / title
        title = QLabel("🍾  BottleVision")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 800; color: {ACCENT}; padding: 4px 0;"
        )
        left_layout.addWidget(title)
        subtitle = QLabel("Liquid Level Estimation System")
        subtitle.setStyleSheet(f"font-size: 11px; color: {FG_DIM}; margin-bottom: 4px;")
        left_layout.addWidget(subtitle)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2a2a4a;")
        left_layout.addWidget(sep)

        # ── Source ──
        src_group = _group("  Source")
        sg_lay    = QVBoxLayout(src_group)

        self.source_combo = QComboBox()
        self.source_combo.setStyleSheet(f"""
            QComboBox {{
                background: {BG_DARK}; color: {FG_TEXT};
                border: 1px solid #2a2a4a; border-radius: 5px;
                padding: 6px 10px;
                font-size: 13px;
                min-height: 28px;
            }}
            QComboBox::drop-down {{
                border: none; width: 24px;
                subcontrol-position: center right;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {FG_DIM};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {BG_CARD}; color: {FG_TEXT};
                selection-background-color: {ACCENT}; selection-color: #000;
                border: 1px solid #3a3a5a;
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 30px;
                padding: 4px 8px;
            }}
        """)
        # Use QStyle standard icons for the dropdown items
        style = self.style()
        video_icon = style.standardIcon(QApplication.style().SP_MediaPlay)
        camera_icon = style.standardIcon(QApplication.style().SP_DriveNetIcon)
        self.source_combo.addItem(video_icon, "  Video File")
        self.source_combo.addItem(camera_icon, "  Camera")
        self.source_combo.setIconSize(QSize(20, 20))
        sg_lay.addWidget(self.source_combo)

        src_row = QHBoxLayout()
        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("video file path…")
        self.source_path.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_DARK}; color: {FG_TEXT};
                border: 1px solid #2a2a4a; border-radius: 5px; padding: 4px 8px;
            }}
        """)
        self.browse_btn = _btn("Browse", "#2a2a5a", FG_TEXT, 70)
        src_row.addWidget(self.source_path)
        src_row.addWidget(self.browse_btn)
        sg_lay.addLayout(src_row)

        self.cam_spin = _spin(0, 10, 0, " cam")
        self.cam_spin.setVisible(False)
        sg_lay.addWidget(self.cam_spin)

        # Model path
        model_row = QHBoxLayout()
        self.model_path_edit = QLineEdit(DEFAULT_MODEL_PATH)
        self.model_path_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_DARK}; color: {FG_TEXT};
                border: 1px solid #2a2a4a; border-radius: 5px; padding: 4px 8px;
            }}
        """)
        self.model_browse_btn = _btn("Model", "#2a2a5a", FG_TEXT, 62)
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(self.model_browse_btn)
        sg_lay.addLayout(model_row)

        left_layout.addWidget(src_group)

        # ── Playback ──
        pb_group = _group("  Playback")
        pb_lay   = QVBoxLayout(pb_group)
        btn_row  = QHBoxLayout()
        btn_row.setSpacing(6)
        self.start_btn = _btn("▶ Start",  ACCENT,   "#000", 90)
        self.pause_btn = _btn("⏸ Pause",  "#3a3a6a", FG_TEXT, 90)
        self.stop_btn  = _btn("⏹ Stop",   RED_COL,   "#fff", 80)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)
        pb_lay.addLayout(btn_row)
        self.show_ids_chk = QCheckBox("Show track IDs")
        self.show_ids_chk.setChecked(True)
        self.show_ids_chk.setStyleSheet(f"color: {FG_TEXT}; font-size: 12px;")
        pb_lay.addWidget(self.show_ids_chk)
        left_layout.addWidget(pb_group)

        # ── Calibration ──
        cal_group = _group("  Calibration  (↑ ↓  keys)")
        cal_lay   = QVBoxLayout(cal_group)

        # Current Y display
        y_row = QHBoxLayout()
        y_row.addWidget(_label("Target line Y:", dim=True))
        self.line_y_display = QLabel(str(self._target_y))
        self.line_y_display.setStyleSheet(
            f"color: {ACCENT}; font-size: 18px; font-weight: 800; padding: 0 6px;"
        )
        y_row.addWidget(self.line_y_display)
        y_row.addStretch()
        cal_lay.addLayout(y_row)

        # Arrow buttons
        arrow_row = QHBoxLayout()
        self.up_btn   = _btn("▲  Up",    "#1f3a2a", ACCENT, 120)
        self.down_btn = _btn("▼  Down",  "#1f3a2a", ACCENT, 120)
        arrow_row.addWidget(self.up_btn)
        arrow_row.addWidget(self.down_btn)
        cal_lay.addLayout(arrow_row)

        # Step size
        step_row = QHBoxLayout()
        step_row.addWidget(_label("Step size:", dim=True))
        self.step_spin = _spin(1, 100, 1, " px")
        self.step_spin.setMaximumWidth(100)
        step_row.addWidget(self.step_spin)
        step_row.addStretch()
        cal_lay.addLayout(step_row)

        # Jump
        jump_row = QHBoxLayout()
        self.jump_spin = _spin(1, 9999, self._target_y)
        self.jump_spin.setToolTip("Type a Y value and press Set to jump there")
        self.jump_btn  = _btn("↳ Set", "#2a2a5a", FG_TEXT, 48)
        jump_row.addWidget(_label("Jump to:", dim=True))
        jump_row.addWidget(self.jump_spin)
        jump_row.addWidget(self.jump_btn)
        cal_lay.addLayout(jump_row)

        # Save calibration
        self.save_cal_btn = _btn("💾  Save Calibration", "#1a3a2a", ACCENT)
        cal_lay.addWidget(self.save_cal_btn)

        cal_note = _label("Shift+↑/↓ = ×10 step", dim=True)
        cal_note.setAlignment(Qt.AlignCenter)
        cal_lay.addWidget(cal_note)
        left_layout.addWidget(cal_group)

        # ── Detection Settings ──
        det_group = _group("  Detection Settings")
        det_lay   = QVBoxLayout(det_group)

        det_defs = [
            ("Tolerance",    "_tol_spin",   lambda: _spin(0, 100, DEFAULT_TOLERANCE, " px")),
            ("Confidence",   "_conf_spin",  lambda: _dspin(0.01, 1.0, DEFAULT_CONF)),
            ("IoU",          "_iou_spin",   lambda: _dspin(0.01, 1.0, DEFAULT_IOU)),
            ("Gradient thr", "_grad_spin",  lambda: _spin(1, 100, DEFAULT_GRADIENT_THR)),
            ("Gauss kernel", "_gauss_spin", lambda: _spin(1, 21, DEFAULT_GAUSS_KERNEL)),
            ("Detect line",  "_dline_spin", lambda: _dspin(0.01, 0.99, DEFAULT_DETECT_LINE)),
        ]
        for label_text, attr, make_widget in det_defs:
            row = QHBoxLayout()
            row.addWidget(_label(label_text + ":", dim=True))
            widget = make_widget()
            widget.setMaximumWidth(110)
            setattr(self, attr, widget)
            row.addStretch()
            row.addWidget(widget)
            det_lay.addLayout(row)

        left_layout.addWidget(det_group)

        # ── Statistics ──
        stat_group = _group("  Statistics")
        stat_lay   = QVBoxLayout(stat_group)
        self._stat_labels = {}
        for key, label_text, color in [
            ("total_bottles", "Total",    FG_TEXT),
            ("passed",        "Passed",   ACCENT),
            ("rejected_low",  "Rejected", RED_COL),
            ("not_detected",  "Unknown",  YELL_COL),
        ]:
            row = QHBoxLayout()
            row.addWidget(_label(label_text + ":", dim=True))
            lbl = QLabel("0")
            lbl.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 800;")
            lbl.setAlignment(Qt.AlignRight)
            self._stat_labels[key] = lbl
            row.addStretch()
            row.addWidget(lbl)
            stat_lay.addLayout(row)

        self.reset_stats_btn = _btn("↺  Reset Stats", "#2a2a4a", FG_DIM)
        stat_lay.addWidget(self.reset_stats_btn)
        left_layout.addWidget(stat_group)
        left_layout.addStretch()

        # ── Center: preview ─────────────────────────────────────────────────
        self.preview = PreviewLabel()

        # ── Right: log ──────────────────────────────────────────────────────
        log_panel = QWidget()
        log_panel.setFixedWidth(320)
        log_panel.setStyleSheet(f"background: {BG_PANEL}; border-radius: 10px;")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(8, 8, 8, 8)

        log_title = QLabel("📋  Event Log")
        log_title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {ACCENT}; margin-bottom: 4px;"
        )
        log_layout.addWidget(log_title)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {BG_DARK};
                color: {FG_TEXT};
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        log_layout.addWidget(self.log_view)
        clear_log_btn = _btn("Clear Log", "#2a2a4a", FG_DIM, 80)
        clear_log_btn.clicked.connect(self.log_view.clear)
        log_layout.addWidget(clear_log_btn, alignment=Qt.AlignRight)

        # ── Assemble ────────────────────────────────────────────────────────
        left_scroll.setWidget(left_panel)
        root.addWidget(left_scroll)
        root.addWidget(self.preview, stretch=1)
        root.addWidget(log_panel)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(
            f"background: {BG_PANEL}; color: {FG_DIM}; font-size: 11px;"
        )
        self.status_bar.showMessage("Ready  —  Select a source and press Start")

    # ── connections ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._browse_video)
        self.model_browse_btn.clicked.connect(self._browse_model)
        self.source_combo.currentIndexChanged.connect(self._source_type_changed)
        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)
        self.up_btn.clicked.connect(lambda: self._move_line(-self.step_spin.value()))
        self.down_btn.clicked.connect(lambda: self._move_line(+self.step_spin.value()))
        self.jump_btn.clicked.connect(self._jump_line)
        self.save_cal_btn.clicked.connect(self._save_calibration)
        self.reset_stats_btn.clicked.connect(self._reset_stats)

    # ── source type ─────────────────────────────────────────────────────────

    def _source_type_changed(self, idx):
        is_file = (idx == 0)
        self.source_path.setVisible(is_file)
        self.browse_btn.setVisible(is_file)
        self.cam_spin.setVisible(not is_file)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
        )
        if path:
            self.source_path.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "",
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    # ── playback ────────────────────────────────────────────────────────────

    def _get_source(self):
        if self.source_combo.currentIndex() == 0:
            return self.source_path.text().strip() or None
        return str(self.cam_spin.value())

    def _start(self):
        src = self._get_source()
        if not src:
            QMessageBox.warning(self, "No Source", "Please select a video file or camera.")
            return
        model = self.model_path_edit.text().strip()
        if not os.path.exists(model):
            QMessageBox.warning(self, "Model Not Found",
                                f"Cannot find model file:\n{model}")
            return
        self._stop()   # stop any existing worker

        self._worker = DetectionWorker(
            source          = src,
            model_path      = model,
            get_target_y    = self._get_target_y,
            conf            = self._conf_spin.value(),
            iou             = self._iou_spin.value(),
            tolerance       = self._tol_spin.value(),
            gauss_kernel    = self._gauss_spin.value(),
            gradient_thr    = self._grad_spin.value(),
            detect_line_frac= self._dline_spin.value(),
            show_ids        = self.show_ids_chk.isChecked(),
        )
        self._worker.frame_ready.connect(self.preview.set_frame, Qt.QueuedConnection)
        self._worker.log_message.connect(self._append_log, Qt.QueuedConnection)
        self._worker.stats_updated.connect(self._update_stats, Qt.QueuedConnection)
        self._worker.error_occurred.connect(self._on_error, Qt.QueuedConnection)
        self._worker.finished_clean.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.status_bar.showMessage("▶  Running…")

    def _toggle_pause(self):
        if self._worker is None:
            return
        if self._worker._paused:
            self._worker.resume()
            self.pause_btn.setText("⏸  Pause")
            self.status_bar.showMessage("▶  Running…")
        else:
            self._worker.pause()
            self.pause_btn.setText("▶  Resume")
            self.status_bar.showMessage("⏸  Paused")

    def _stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸  Pause")
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("⏹  Stopped")

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("⏹  Finished")

    def _on_error(self, msg):
        self._append_log(f"❌  ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)
        self._stop()

    # ── virtual line calibration ─────────────────────────────────────────────

    def _get_target_y(self):
        with self._target_y_lock:
            return self._target_y

    def _set_target_y(self, y):
        with self._target_y_lock:
            self._target_y = max(0, y)
        self.line_y_display.setText(str(self._target_y))
        self.jump_spin.setValue(self._target_y)

    def _move_line(self, delta):
        self._set_target_y(self._get_target_y() + delta)

    def _jump_line(self):
        self._set_target_y(self.jump_spin.value())

    def _save_calibration(self):
        y = self._get_target_y()
        out = f"TARGET_LINE_Y = {y}  # pixels from top of frame\n"
        self._append_log(f"💾  Calibration saved — TARGET_LINE_Y = {y}")
        try:
            with open("calibration.txt", "w") as f:
                f.write(f"# BottleVision Calibration\n")
                f.write(f"TARGET_LINE_Y = {y}\n")
            self.status_bar.showMessage(f"Calibration saved to calibration.txt  (Y={y})")
        except Exception as e:
            self._append_log(f"⚠  Could not write file: {e}")

    # ── stats ────────────────────────────────────────────────────────────────

    def _update_stats(self, stats: dict):
        self._last_stats = stats
        for key, lbl in self._stat_labels.items():
            lbl.setText(str(stats.get(key, 0)))

    def _reset_stats(self):
        for lbl in self._stat_labels.values():
            lbl.setText("0")
        self._append_log("↺  Statistics reset")

    # ── log ─────────────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── keyboard ─────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key  = event.key()
        mods = event.modifiers()
        step = self.step_spin.value()
        if mods & Qt.ShiftModifier:
            step *= 10

        if key == Qt.Key_Up:
            self._move_line(-step)
            event.accept()
        elif key == Qt.Key_Down:
            self._move_line(+step)
            event.accept()
        elif key == Qt.Key_Space:
            self._toggle_pause()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop()
        event.accept()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BottleVision")
    app.setApplicationVersion("1.0.0")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
