#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bottle Liquid Level Inspection System
Professional PyQt-based GUI for automated liquid level quality control
Using YOLOv8 + ByteTrack for real-time bottle detection and liquid level estimation
"""

import sys
import os
import json
import csv
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QFileDialog, QTextEdit, QGroupBox, QSpinBox, 
                             QDoubleSpinBox, QCheckBox, QStatusBar, QMenuBar,
                             QMenu, QAction, QSplitter, QFrame, QScrollArea,
                             QComboBox, QMessageBox, QProgressDialog, QShortcut,
                             QInputDialog, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence, QIcon
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================
DEFAULT_CONFIG = {
    'yolo_model': 'best.pt',
    'target_line_y': 730,
    'tolerance': 5,
    'detection_line_x_percent': 0.5,
    'conf_threshold': 0.3,
    'iou_threshold': 0.45,
    'gaussian_kernel': 5,
    'min_gradient_threshold': 10,
    'display_scale': 0.7,
    'show_debug': False,
    'show_tracking_ids': True
}


# =============================================================================
# LIQUID LEVEL DETECTION ENGINE
# =============================================================================
class LiquidLevelDetector:
    """Core liquid level detection using computer vision"""
    
    @staticmethod
    def detect_liquid_level(gray_img, min_gradient_threshold=10):
        """
        Detect liquid level in grayscale image of liquid/air interface
        
        Args:
            gray_img: Grayscale image showing liquid/air interface
            min_gradient_threshold: Minimum gradient value to be considered valid
        
        Returns:
            liquid_level: Y position of detected liquid level (or None if not found)
            gradient_info: Dictionary with gradient analysis details
        """
        if gray_img.size == 0 or gray_img.shape[0] < 2:
            return None, {}
        
        # Calculate vertical intensity profile (average across width)
        intensity_profile = np.mean(gray_img, axis=1)
        
        # Find the steepest gradient (liquid/air transition)
        gradients = np.diff(intensity_profile)
        abs_gradients = np.abs(gradients)
        
        if len(abs_gradients) == 0:
            return None, {}
        
        # Find strongest gradient
        max_gradient_idx = np.argmax(abs_gradients)
        max_gradient_value = abs_gradients[max_gradient_idx]
        
        # Verify we have a significant transition
        if max_gradient_value < min_gradient_threshold:
            # Fallback: Look for consistent dark region
            dark_threshold = np.mean(intensity_profile) * 0.7
            liquid_region = np.where(intensity_profile < dark_threshold)[0]
            
            if len(liquid_region) > 0:
                liquid_level = liquid_region[0]
            else:
                return None, {
                    'intensity_profile': intensity_profile,
                    'gradients': gradients,
                    'max_gradient_idx': max_gradient_idx,
                    'max_gradient_value': max_gradient_value
                }
        else:
            liquid_level = max_gradient_idx + 1
        
        return liquid_level, {
            'intensity_profile': intensity_profile,
            'gradients': gradients,
            'max_gradient_idx': max_gradient_idx,
            'max_gradient_value': max_gradient_value
        }
    
    @staticmethod
    def process_roi(roi_img, gaussian_kernel=5, min_gradient=10):
        """
        Process ROI with Gaussian blur and detect liquid level
        
        Returns:
            liquid_level_relative: Liquid level position relative to ROI top (pixels)
            processed_img: Processed grayscale image for visualization
        """
        if roi_img.size == 0:
            return None, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
        
        # Detect liquid level
        liquid_level_relative, _ = LiquidLevelDetector.detect_liquid_level(
            blurred, min_gradient_threshold=min_gradient
        )
        
        return liquid_level_relative, blurred


# =============================================================================
# BOTTLE TRACKER WITH YOLO + BYTETRACK
# =============================================================================
class BottleTracker:
    """YOLOv8 + ByteTrack based bottle tracking and liquid level estimation"""
    
    def __init__(self, model_path, detection_line_x, target_line_y, tolerance):
        self.model = YOLO(model_path) if os.path.exists(model_path) else None
        self.detection_line_x = detection_line_x
        self.target_line_y = target_line_y
        self.tolerance = tolerance
        
        # Tracking state
        self.processed_bottles = set()
        self.bottle_results = {}
        
        # Statistics
        self.stats = {
            'total_bottles': 0,
            'passed': 0,
            'rejected_low': 0,
            'not_detected': 0
        }
    
    def has_crossed_line(self, bbox, track_id):
        """Check if bottle has crossed the detection line"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        if center_x >= self.detection_line_x and track_id not in self.processed_bottles:
            return True
        return False
    
    def classify_bottle(self, liquid_level_absolute):
        """Classify bottle as PASS or REJECT"""
        if liquid_level_absolute is None:
            return "NOT_DETECTED"
        
        deviation = liquid_level_absolute - self.target_line_y
        
        if deviation <= self.tolerance:
            return "PASS"
        else:
            return "REJECT_LOW"
    
    def process_bottle(self, frame, bbox, track_id, config):
        """Process a bottle that has crossed the detection line"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop ROI
        roi = frame[y1:y2, x1:x2]
        
        # Process ROI
        liquid_level_relative, processed_img = LiquidLevelDetector.process_roi(
            roi, 
            gaussian_kernel=config.get('gaussian_kernel', 5),
            min_gradient=config.get('min_gradient_threshold', 10)
        )
        
        # Calculate absolute liquid level
        liquid_level_absolute = None
        if liquid_level_relative is not None:
            liquid_level_absolute = y1 + liquid_level_relative
        
        # Classify
        status = self.classify_bottle(liquid_level_absolute)
        
        # Store results
        result = {
            'track_id': track_id,
            'bbox': (x1, y1, x2, y2),
            'liquid_level_relative': liquid_level_relative,
            'liquid_level_absolute': liquid_level_absolute,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        self.bottle_results[track_id] = result
        self.processed_bottles.add(track_id)
        
        # Update statistics
        self.stats['total_bottles'] += 1
        if status == 'PASS':
            self.stats['passed'] += 1
        elif status == 'REJECT_LOW':
            self.stats['rejected_low'] += 1
        else:
            self.stats['not_detected'] += 1
        
        return result
    
    def detect_and_track(self, frame, config):
        """Run YOLO detection with ByteTrack tracking"""
        if self.model is None:
            return None, []
        
        try:
            results = self.model.track(
                frame,
                conf=config.get('conf_threshold', 0.3),
                iou=config.get('iou_threshold', 0.45),
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                imgsz=640
            )
            
            newly_processed = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    
                    for bbox, track_id in zip(boxes, track_ids):
                        if self.has_crossed_line(bbox, track_id):
                            result_dict = self.process_bottle(frame, bbox, track_id, config)
                            newly_processed.append(result_dict)
            
            return results, newly_processed
        
        except Exception as e:
            print(f"Detection error: {e}")
            return None, []
    
    def visualize(self, frame, results):
        """Visualize tracking, detection line, and liquid levels"""
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw target line (GREEN)
        cv2.line(vis_frame, (0, self.target_line_y), (width, self.target_line_y), 
                (0, 255, 0), 2)
        cv2.putText(vis_frame, "TARGET", (10, self.target_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tolerance bounds (RED dashed)
        for y in [self.target_line_y - self.tolerance, self.target_line_y + self.tolerance]:
            for i in range(0, width, 20):
                cv2.line(vis_frame, (i, y), (min(i+10, width), y), (0, 0, 255), 1)
        
        # Draw detection line (CYAN vertical)
        cv2.line(vis_frame, (self.detection_line_x, 0), (self.detection_line_x, height),
                (255, 255, 0), 2)
        
        # Draw YOLO detections
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for bbox, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Determine color based on status
                    if track_id in self.bottle_results:
                        result_data = self.bottle_results[track_id]
                        status = result_data['status']
                        
                        if status == 'PASS':
                            color = (0, 255, 0)
                        elif status == 'REJECT_LOW':
                            color = (0, 0, 255)
                        else:
                            color = (0, 165, 255)
                        
                        # Draw liquid level line
                        liquid_level = result_data['liquid_level_absolute']
                        if liquid_level is not None:
                            cv2.line(vis_frame, (x1, int(liquid_level)), (x2, int(liquid_level)),
                                   color, 2)
                    else:
                        color = (255, 255, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track ID
                    label = f"ID:{track_id}"
                    if track_id in self.bottle_results:
                        label += f" {self.bottle_results[track_id]['status']}"
                    
                    cv2.putText(vis_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame


# =============================================================================
# VIDEO PROCESSING THREAD
# =============================================================================
class VideoProcessor(QThread):
    """Background thread for video processing"""
    frame_ready = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_source, tracker, config):
        super().__init__()
        self.video_source = video_source
        self.tracker = tracker
        self.config = config
        self.running = True
        self.paused = False
        self.cap = None
    
    def run(self):
        """Main processing loop"""
        try:
            # Open video source
            try:
                source = int(self.video_source)
                self.cap = cv2.VideoCapture(source)
            except ValueError:
                self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Cannot open video source: {self.video_source}")
                return
            
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 24
            frame_delay = 1.0 / fps
            
            while self.running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.tracker.processed_bottles.clear()
                        continue
                    
                    # Process frame
                    results, newly_processed = self.tracker.detect_and_track(frame, self.config)
                    
                    # Visualize
                    vis_frame = self.tracker.visualize(frame, results)
                    
                    # Emit frame
                    self.frame_ready.emit(vis_frame)
                    
                    # Emit stats periodically
                    self.stats_updated.emit(self.tracker.stats.copy())
                    
                    # Log new detections
                    for result in newly_processed:
                        self.log_message.emit(
                            f"Bottle #{result['track_id']}: {result['status']} "
                            f"(Level: Y={result['liquid_level_absolute']})"
                        )
                    
                    # Maintain FPS
                    QThread.msleep(int(frame_delay * 1000))
                else:
                    QThread.msleep(100)
        
        except Exception as e:
            self.error_occurred.emit(str(e))
        
        finally:
            if self.cap:
                self.cap.release()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def stop(self):
        self.running = False
        self.wait()


# =============================================================================
# MAIN APPLICATION WINDOW
# =============================================================================
class BottleLiquidInspector(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.config = DEFAULT_CONFIG.copy()
        self.tracker = None
        self.processor = None
        self.current_video = None
        
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Bottle Liquid Level Inspection System")
        self.setMinimumSize(1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Top toolbar
        self.create_toolbar()
        
        # Main content area
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Video preview
        left_panel = self.create_video_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Controls
        right_panel = self.create_control_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Keyboard shortcuts
        self.setup_shortcuts()
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        load_video_action = QAction('&Load Video', self)
        load_video_action.setShortcut('Ctrl+O')
        load_video_action.triggered.connect(self.load_video)
        file_menu.addAction(load_video_action)
        
        load_camera_action = QAction('&Use Camera', self)
        load_camera_action.setShortcut('Ctrl+C')
        load_camera_action.triggered.connect(self.use_camera)
        file_menu.addAction(load_camera_action)
        
        file_menu.addSeparator()
        
        save_config_action = QAction('&Save Configuration', self)
        save_config_action.setShortcut('Ctrl+S')
        save_config_action.triggered.connect(self.save_config_file)
        file_menu.addAction(save_config_action)
        
        load_config_action = QAction('&Load Configuration', self)
        load_config_action.setShortcut('Ctrl+L')
        load_config_action.triggered.connect(self.load_config_file)
        file_menu.addAction(load_config_action)
        
        file_menu.addSeparator()
        
        export_results_action = QAction('&Export Results (CSV)', self)
        export_results_action.setShortcut('Ctrl+E')
        export_results_action.triggered.connect(self.export_results)
        file_menu.addAction(export_results_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create top toolbar"""
        toolbar = QHBoxLayout()
        
        # Video source info
        self.video_label = QLabel("No video loaded")
        self.video_label.setMinimumWidth(200)
        toolbar.addWidget(self.video_label)
        
        toolbar.addStretch()
        
        # Playback controls
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        toolbar.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        toolbar.addWidget(self.stop_btn)
        
        # Processing stats
        self.fps_label = QLabel("FPS: 0")
        toolbar.addWidget(self.fps_label)
        
        self.bottles_label = QLabel("Bottles: 0")
        toolbar.addWidget(self.bottles_label)
        
        self.pass_label = QLabel("✓ Pass: 0")
        self.pass_label.setStyleSheet("color: green")
        toolbar.addWidget(self.pass_label)
        
        self.reject_label = QLabel("✗ Reject: 0")
        self.reject_label.setStyleSheet("color: red")
        toolbar.addWidget(self.reject_label)
        
        # Add toolbar to main layout
        central_widget = self.centralWidget()
        main_layout = central_widget.layout()
        main_layout.insertLayout(0, toolbar)
    
    def create_video_panel(self):
        """Create video preview panel"""
        panel = QGroupBox("Video Preview")
        layout = QVBoxLayout(panel)
        
        # Video display label
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("background-color: #000;")
        layout.addWidget(self.video_display)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        zoom_layout.addWidget(self.zoom_label)
        
        layout.addLayout(zoom_layout)
        
        return panel
    
    def create_control_panel(self):
        """Create control panel with all settings"""
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Calibration section
        calib_group = QGroupBox("Virtual Line Calibration")
        calib_layout = QVBoxLayout(calib_group)
        
        # Target line Y position
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Line Y:"))
        self.target_y_spin = QSpinBox()
        self.target_y_spin.setRange(0, 4096)
        self.target_y_spin.setValue(self.config['target_line_y'])
        self.target_y_spin.valueChanged.connect(self.update_target_line)
        target_layout.addWidget(self.target_y_spin)
        calib_layout.addLayout(target_layout)
        
        # Fine adjustment buttons
        fine_layout = QHBoxLayout()
        fine_layout.addWidget(QLabel("Fine Adjust:"))
        
        up_btn = QPushButton("↑ Up 1px")
        up_btn.clicked.connect(lambda: self.adjust_target_line(-1))
        fine_layout.addWidget(up_btn)
        
        down_btn = QPushButton("↓ Down 1px")
        down_btn.clicked.connect(lambda: self.adjust_target_line(1))
        fine_layout.addWidget(down_btn)
        
        calib_layout.addLayout(fine_layout)
        
        # Coarse adjustment buttons
        coarse_layout = QHBoxLayout()
        coarse_layout.addWidget(QLabel("Coarse Adjust:"))
        
        up_coarse_btn = QPushButton("↑↑ Up 10px")
        up_coarse_btn.clicked.connect(lambda: self.adjust_target_line(-10))
        coarse_layout.addWidget(up_coarse_btn)
        
        down_coarse_btn = QPushButton("↓↓ Down 10px")
        down_coarse_btn.clicked.connect(lambda: self.adjust_target_line(10))
        coarse_layout.addWidget(down_coarse_btn)
        
        calib_layout.addLayout(coarse_layout)
        
        # Tolerance setting
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Tolerance (±px):"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(1, 50)
        self.tolerance_spin.setValue(self.config['tolerance'])
        self.tolerance_spin.valueChanged.connect(self.update_tolerance)
        tol_layout.addWidget(self.tolerance_spin)
        calib_layout.addLayout(tol_layout)
        
        layout.addWidget(calib_group)
        
        # Detection settings
        detect_group = QGroupBox("Detection Settings")
        detect_layout = QVBoxLayout(detect_group)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.config['conf_threshold'])
        conf_layout.addWidget(self.conf_spin)
        detect_layout.addLayout(conf_layout)
        
        # Gaussian kernel
        gauss_layout = QHBoxLayout()
        gauss_layout.addWidget(QLabel("Gaussian Kernel:"))
        self.gauss_spin = QSpinBox()
        self.gauss_spin.setRange(1, 15)
        self.gauss_spin.setSingleStep(2)
        self.gauss_spin.setValue(self.config['gaussian_kernel'])
        gauss_layout.addWidget(self.gauss_spin)
        detect_layout.addLayout(gauss_layout)
        
        # Gradient threshold
        grad_layout = QHBoxLayout()
        grad_layout.addWidget(QLabel("Min Gradient:"))
        self.grad_spin = QSpinBox()
        self.grad_spin.setRange(1, 50)
        self.grad_spin.setValue(self.config['min_gradient_threshold'])
        grad_layout.addWidget(self.grad_spin)
        detect_layout.addLayout(grad_layout)
        
        layout.addWidget(detect_group)
        
        # Model selection
        model_group = QGroupBox("YOLO Model")
        model_layout = QVBoxLayout(model_group)
        
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit(self.config['yolo_model'])
        model_path_layout.addWidget(self.model_path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_path_layout.addWidget(browse_btn)
        
        model_layout.addLayout(model_path_layout)
        layout.addWidget(model_group)
        
        # Reset button
        reset_btn = QPushButton("Reset Statistics")
        reset_btn.clicked.connect(self.reset_statistics)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        
        panel.setWidget(container)
        return panel
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Arrow keys for fine adjustment
        QShortcut(QKeySequence(Qt.Key_Up), self, self.on_arrow_up)
        QShortcut(QKeySequence(Qt.Key_Down), self, self.on_arrow_down)
        
        # Shift + Arrow for coarse adjustment
        QShortcut(QKeySequence("Shift+Up"), self, self.on_shift_arrow_up)
        QShortcut(QKeySequence("Shift+Down"), self, self.on_shift_arrow_down)
        
        # Space for play/pause
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_playback)
    
    def on_arrow_up(self):
        """Arrow up - fine adjustment (-1px)"""
        if self.processor and not self.processor.paused:
            self.adjust_target_line(-1)
    
    def on_arrow_down(self):
        """Arrow down - fine adjustment (+1px)"""
        if self.processor and not self.processor.paused:
            self.adjust_target_line(1)
    
    def on_shift_arrow_up(self):
        """Shift+Arrow up - coarse adjustment (-10px)"""
        if self.processor and not self.processor.paused:
            self.adjust_target_line(-10)
    
    def on_shift_arrow_down(self):
        """Shift+Arrow down - coarse adjustment (+10px)"""
        if self.processor and not self.processor.paused:
            self.adjust_target_line(10)
    
    def adjust_target_line(self, delta):
        """Adjust target line position by delta pixels"""
        current = self.target_y_spin.value()
        new_value = max(0, min(4096, current + delta))
        self.target_y_spin.setValue(new_value)
    
    def update_target_line(self, value):
        """Update target line Y position"""
        self.config['target_line_y'] = value
        self.statusBar.showMessage(f"Target line updated: Y={value}")
    
    def update_tolerance(self, value):
        """Update tolerance value"""
        self.config['tolerance'] = value
        self.statusBar.showMessage(f"Tolerance updated: ±{value}px")
    
    def update_zoom(self, value):
        """Update video zoom level"""
        self.zoom_label.setText(f"{value}%")
    
    def load_video(self):
        """Load video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.current_video = file_path
            self.video_label.setText(f"Video: {Path(file_path).name}")
            self.start_processing()
    
    def use_camera(self):
        """Use camera feed"""
        camera_num, ok = QInputDialog.getInt(
            self,
            "Select Camera",
            "Enter camera index (usually 0 for default webcam):",
            0, 0, 10, 1
        )
        
        if ok:
            self.current_video = str(camera_num)
            self.video_label.setText(f"Camera {camera_num}")
            self.start_processing()
    
    def start_processing(self):
        """Start video processing"""
        if not self.current_video:
            return
        
        # Initialize tracker
        model_path = self.model_path_edit.text()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Not Found", f"YOLO model not found: {model_path}")
            return
        
        width = 1280  # Default width
        detection_line_x = int(width * self.config['detection_line_x_percent'])
        
        self.tracker = BottleTracker(
            model_path=model_path,
            detection_line_x=detection_line_x,
            target_line_y=self.config['target_line_y'],
            tolerance=self.config['tolerance']
        )
        
        # Start processor thread
        self.processor = VideoProcessor(self.current_video, self.tracker, self.config)
        self.processor.frame_ready.connect(self.display_frame)
        self.processor.stats_updated.connect(self.update_stats)
        self.processor.log_message.connect(self.add_log_message)
        self.processor.error_occurred.connect(self.show_error)
        self.processor.start()
        
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.play_btn.setText("⏸ Pause")
    
    def stop_processing(self):
        """Stop video processing"""
        if self.processor:
            self.processor.stop()
            self.processor = None
        
        self.play_btn.setText("▶ Play")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
    
    def toggle_playback(self):
        """Toggle playback pause/resume"""
        if not self.processor:
            return
        
        if self.processor.paused:
            self.processor.resume()
            self.play_btn.setText("⏸ Pause")
        else:
            self.processor.pause()
            self.play_btn.setText("▶ Play")
    
    def display_frame(self, frame):
        """Display video frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get current zoom
        zoom_factor = self.zoom_slider.value() / 100.0
        if zoom_factor != 1.0:
            h, w = rgb_frame.shape[:2]
            new_size = (int(w * zoom_factor), int(h * zoom_factor))
            rgb_frame = cv2.resize(rgb_frame, new_size)
        
        # Create QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Set pixmap
        pixmap = QPixmap.fromImage(qimg)
        self.video_display.setPixmap(pixmap.scaled(
            self.video_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
    
    def update_stats(self, stats):
        """Update statistics display"""
        self.bottles_label.setText(f"Bottles: {stats['total_bottles']}")
        self.pass_label.setText(f"✓ Pass: {stats['passed']}")
        self.reject_label.setText(f"✗ Reject: {stats['rejected_low']}")
    
    def add_log_message(self, message):
        """Add message to log"""
        # Could add a dedicated log panel here
        self.statusBar.showMessage(message, 5000)
    
    def show_error(self, error_msg):
        """Show error message"""
        QMessageBox.critical(self, "Error", error_msg)
    
    def reset_statistics(self):
        """Reset tracking statistics"""
        if self.tracker:
            self.tracker.stats = {
                'total_bottles': 0,
                'passed': 0,
                'rejected_low': 0,
                'not_detected': 0
            }
            self.tracker.processed_bottles.clear()
            self.tracker.bottle_results.clear()
            self.statusBar.showMessage("Statistics reset!")
    
    def save_config_file(self):
        """Save configuration to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "config.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            self.config['yolo_model'] = self.model_path_edit.text()
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.statusBar.showMessage(f"Configuration saved to {file_path}")
    
    def load_config_file(self):
        """Load configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            with open(file_path, 'r') as f:
                loaded_config = json.load(f)
            
            self.config.update(loaded_config)
            self.apply_config()
            self.statusBar.showMessage(f"Configuration loaded from {file_path}")
    
    def load_config(self):
        """Load default or existing configuration"""
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
                self.apply_config()
    
    def apply_config(self):
        """Apply configuration to UI controls"""
        self.target_y_spin.setValue(self.config['target_line_y'])
        self.tolerance_spin.setValue(self.config['tolerance'])
        self.conf_spin.setValue(self.config['conf_threshold'])
        self.gauss_spin.setValue(self.config['gaussian_kernel'])
        self.grad_spin.setValue(self.config['min_gradient_threshold'])
        self.model_path_edit.setText(self.config['yolo_model'])
    
    def export_results(self):
        """Export results to CSV"""
        if not self.tracker or not self.tracker.bottle_results:
            QMessageBox.information(self, "No Data", "No results to export yet!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Track ID', 'Timestamp', 'Liquid Level Y', 'Status', 
                               'Deviation', 'Bounding Box'])
                
                for track_id, result in sorted(self.tracker.bottle_results.items()):
                    deviation = (result['liquid_level_absolute'] - self.config['target_line_y'] 
                               if result['liquid_level_absolute'] else 'N/A')
                    bbox = result['bbox']
                    
                    writer.writerow([
                        track_id,
                        result['timestamp'],
                        result['liquid_level_absolute'] or 'N/A',
                        result['status'],
                        deviation,
                        f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[2]})"
                    ])
            
            self.statusBar.showMessage(f"Results exported to {file_path}")
    
    def browse_model(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "PyTorch Models (*.pt)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Bottle Liquid Inspector",
            "<h2>Bottle Liquid Level Inspection System</h2>"
            "<p>Version 1.0.0</p>"
            "<p>Professional automated liquid level quality control system</p>"
            "<p>Using YOLOv8 + ByteTrack for real-time detection and tracking</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Real-time video processing</li>"
            "<li>Precision liquid level measurement</li>"
            "<li>Pass/Fail classification</li>"
            "<li>Statistical reporting</li>"
            "<li>Configurable parameters</li>"
            "</ul>"
        )
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.processor:
            self.processor.stop()
        event.accept()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = BottleLiquidInspector()
    window.show()
    
    sys.exit(app.exec_())
