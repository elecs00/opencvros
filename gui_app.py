# gui_app.py

import sys
import os
import glob
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QLineEdit, QFileDialog, QMessageBox, QFrame, QComboBox, QGroupBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QMouseEvent, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import numpy as np
from scipy.spatial import distance as dist
import json
from PIL import Image, ImageTk
from PIL.ExifTags import TAGS
from config_manager import ConfigManager, get_mediapipe_config

import visualizer
import cv2
import time
import torch
import argparse

try:
    import rclpy
    from Ros2_ws.src.result_publisher.result_publisher.publisher_node import ImagePublisher
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ROS2 Python 패키지 상대경로 자동 추가 (sleep 프로젝트 어디서든 동작)
ROOT = os.path.dirname(os.path.abspath(__file__))
site_packages_glob = os.path.join(ROOT, 'Ros2_ws', 'install', '*', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
for path in glob.glob(site_packages_glob):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Add root directory to path for imports
if ROOT not in sys.path:
    sys.path.append(ROOT)

import mediapipe_analyzer # Ensure mediapipe_analyzer.py is in the same directory or accessible

from mediapipe.tasks.python import vision
import mediapipe as mp
from mediapipe import Image as mp_Image

from detector_utils import calculate_ear, calculate_mar

GUI_STATE_FILE = "gui_state.json"

def get_exif_orientation(image_path):
    """EXIF Orientation 정보를 읽어서 회전 방향을 반환"""
    # 비디오 파일인 경우 EXIF 읽기 시도하지 않음
    video_extensions = ['.mov', '.mp4', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    file_ext = Path(image_path).suffix.lower()
    
    if file_ext in video_extensions:
        print(f"[EXIF] Video file detected, no rotation applied")
        return 1  # 회전 없음 (기본값)
    
    # 이미지 파일인 경우 EXIF 정보 읽기 시도
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif:
                for tag_id in exif:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exif.get(tag_id)
                    if tag == 'Orientation':
                        print(f"[EXIF] Image orientation: {data}")
                        return data
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
    
    print(f"[EXIF] No rotation applied (default)")
    return 1  # 기본값 (회전 없음)

def apply_exif_rotation(frame, orientation):
    """EXIF Orientation에 따라 프레임을 회전"""
    if orientation == 1:
        return frame  # 정상
    elif orientation == 2:
        return cv2.flip(frame, 1)  # 좌우 반전
    elif orientation == 3:
        return cv2.rotate(frame, cv2.ROTATE_180)  # 180도 회전
    elif orientation == 4:
        return cv2.flip(frame, 0)  # 상하 반전
    elif orientation == 5:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(frame, 1)  # 90도 시계방향 + 좌우 반전
    elif orientation == 6:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 90도 시계방향
    elif orientation == 7:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.flip(frame, 1)  # 90도 반시계방향 + 좌우 반전
    elif orientation == 8:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 90도 반시계방향
    else:
        return frame  # 알 수 없는 값은 그대로 반환

class DragDropLineEdit(QLineEdit):
    """Custom QLineEdit that accepts drag and drop for video and image files"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drag video/image file(s) here or enter source (0 for webcam)")
        self.image_files = []  # Store multiple image files for sequential viewing
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
        
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            # Check if it's a video or image file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
            supported_extensions = video_extensions + image_extensions
            
            # Process all dropped files
            video_files = []
            image_files = []
            
            for url in urls:
                file_path = url.toLocalFile()
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in video_extensions:
                    video_files.append(file_path)
                elif file_ext in image_extensions:
                    image_files.append(file_path)
            
            # Handle the files
            if video_files and image_files:
                QMessageBox.warning(self, "Mixed Files", 
                                   "Please drop either video files OR image files, not both.")
                return
            elif video_files:
                # Single video file - use the first one
                self.setText(video_files[0])
                self.image_files = []
            elif image_files:
                # Multiple image files - store them for sequential viewing
                self.image_files = image_files
                self.setText(f"Image Sequence ({len(image_files)} files)")
            else:
                QMessageBox.warning(self, "Invalid Files", 
                                   f"Please drop video or image files. Supported formats:\n"
                                   f"Video: {', '.join(video_extensions)}\n"
                                   f"Image: {', '.join(image_extensions)}")
        self.setFocus()  # 드롭 후 입력창에 포커스 강제 부여
    
    def get_image_files(self):
        """Return the list of image files for sequential viewing"""
        return self.image_files.copy()
    
    def clear_image_files(self):
        """Clear the image files list"""
        self.image_files = []

    def setText(self, text):
        super().setText(text)
        # 만약 0(웹캠) 또는 비디오 파일이면 image_files 비우기
        if text.strip() == '0' or text.strip().lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
            self.clear_image_files()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, int) # frame, progress_percent
    is_running = False

    def __init__(self, config_args):
        super().__init__()
        self.config_args = config_args
        self.is_running = True
        
        self.visualizer_instance = visualizer.Visualizer() # Visualizer 객체를 여기서 한번만 생성
        self.mediapipe_analyzer = None
        self.openvino_analyzer = None
        self.openvino_hybrid_analyzer = None  # OpenVINO 하이브리드 분석기 추가

        # EXIF 회전 정보 저장
        self.exif_orientation = 1

        # Front Face Calibration related variables
        self._dlib_calibration_trigger = False
        self._mediapipe_calibration_trigger = False
        self._openvino_calibration_trigger = False  # Add OpenVINO calibration trigger
        self._set_dlib_front_face_mode = False
        self._set_mediapipe_front_face_mode = False
        self._set_openvino_front_face_mode = False

        # Live Stream 모드용 결과 저장 변수
        self.latest_face_result = None
        self.latest_hand_result = None

        # --- Driver Status Analysis Variables ---
        self.eye_counter = 0
        self.nod_counter = 0
        self.distract_counter = 0
        self.yawn_counter = 0
        
        self.driver_status = "Normal"
        
        # Image sequence handling
        self.image_files = config_args.get('image_files', [])
        self.current_image_index = 0
        self.is_image_sequence = len(self.image_files) > 0
        # FPS timing
        self.prev_frame_time = None
        
        # --- FPS 제한을 위한 타이밍 제어 ---
        self.target_fps = 20.0
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0
        
        # --- 강력한 FPS 제한을 위한 추가 변수 ---
        self.frame_skip_counter = 0
        self.max_frame_skip = 3  # 최대 3프레임 스킵

        # Crop offset
        self._crop_offset = 0
        self.requested_seek_frame = None  # 안전한 시킹 요청 변수 추가

        # ROS2 Publisher 인스턴스 생성 (앱 전체에서 1회만, 조건부)
        self.ros2_publisher = None
        self.enable_ros2_sending = self.config_args.get('enable_ros2_sending', False)
        if self.enable_ros2_sending and ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            self.ros2_publisher = ImagePublisher()
        elif self.enable_ros2_sending and not ROS2_AVAILABLE:
            print("[Warning] ROS2 Python 패키지가 설치되어 있지 않습니다. ROS2 전송 기능이 비활성화됩니다.")
            self.enable_ros2_sending = False

    # Dlib 캘리브레이션 트리거 getter/setter
    @property
    def dlib_calibration_trigger(self):
        return self._dlib_calibration_trigger

    @dlib_calibration_trigger.setter
    def dlib_calibration_trigger(self, value):
        self._dlib_calibration_trigger = value

    # MediaPipe 캘리브레이션 트리거 getter/setter
    @property
    def mediapipe_calibration_trigger(self):
        return self._mediapipe_calibration_trigger

    @mediapipe_calibration_trigger.setter
    def mediapipe_calibration_trigger(self, value):
        self._mediapipe_calibration_trigger = value

    # OpenVINO 캘리브레이션 트리거 getter/setter
    @property
    def openvino_calibration_trigger(self):
        return self._openvino_calibration_trigger

    @openvino_calibration_trigger.setter
    def openvino_calibration_trigger(self, value):
        self._openvino_calibration_trigger = value
    
    # Dlib Set Front Face Mode getter/setter
    @property
    def set_dlib_front_face_mode(self):
        return self._set_dlib_front_face_mode

    @set_dlib_front_face_mode.setter
    def set_dlib_front_face_mode(self, value):
        self._set_dlib_front_face_mode = value

    # MediaPipe Set Front Face Mode getter/setter
    @property
    def set_mediapipe_front_face_mode(self):
        return self._set_mediapipe_front_face_mode

    @set_mediapipe_front_face_mode.setter
    def set_mediapipe_front_face_mode(self, value):
        self._set_mediapipe_front_face_mode = value

    # OpenVINO Set Front Face Mode getter/setter
    @property
    def set_openvino_front_face_mode(self):
        return self._set_openvino_front_face_mode

    @set_openvino_front_face_mode.setter
    def set_openvino_front_face_mode(self, value):
        self._set_openvino_front_face_mode = value

    # Crop offset getter/setter
    @property
    def crop_offset(self):
        return self._crop_offset

    @crop_offset.setter
    def crop_offset(self, value):
        self._crop_offset = value
        # Visualizer에도 즉시 반영
        if hasattr(self, 'visualizer_instance'):
            self.visualizer_instance.crop_offset = value

    def on_face_result(self, result: 'vision.FaceLandmarkerResult', image: 'mp_Image', timestamp_ms: int):
        self.latest_face_result = result

    def on_hand_result(self, result: 'vision.HandLandmarkerResult', image: 'mp_Image', timestamp_ms: int):
        self.latest_hand_result = result

    def run(self):
        source = self.config_args.get('source', '0')
        imgsz = self.config_args.get('imgsz', 640)
        conf_thres = self.config_args.get('conf_thres', 0.25)
        iou_thres = self.config_args.get('iou_thres', 0.45)
        max_det = self.config_args.get('max_det', 10)
        device = self.config_args.get('device', '')
        hide_labels = self.config_args.get('hide_labels', False)
        hide_conf = self.config_args.get('hide_conf', False)
        half = self.config_args.get('half', False)
        weights = self.config_args.get('weights', os.path.join(ROOT, 'weights', 'last.pt'))
        enable_mediapipe = self.config_args.get('enable_mediapipe', False)
        enable_openvino = self.config_args.get('enable_openvino', False)
        enable_openvino_hybrid = self.config_args.get('enable_openvino_hybrid', False)  # 하이브리드 모드 추가
        # mediapipe_mode_str = self.config_args.get('mediapipe_mode', 'Video (File)')
        
        # Aspect ratio 설정 가져오기
        self.aspect_ratio_mode = self.config_args.get('aspect_ratio', 0)  # 0=Stretch, 1=Fit, 2=Crop, 3=Crop(Top)
        
        # Crop offset 설정 가져오기
        try:
            self._crop_offset = int(self.config_args.get('crop_offset', '0'))
        except ValueError:
            self._crop_offset = 0
        
        # Visualizer에 초기 crop offset 설정
        self.visualizer_instance.crop_offset = self._crop_offset
        
        # 모듈 초기화
        if enable_mediapipe:
            # print(f"[VideoThread] Initializing MediaPipe Analyzer...")
            
            # config.json에서 MediaPipe 모드 설정 읽기
            use_video_mode = get_mediapipe_config("use_video_mode", True)
            
            running_mode = (
                vision.RunningMode.VIDEO if use_video_mode 
                else vision.RunningMode.LIVE_STREAM
            )
            
            mode_str = "VIDEO" if use_video_mode else "LIVE_STREAM"
            # print(f"[VideoThread] MediaPipe mode: {mode_str} (from config.json)")
            
            self.mediapipe_analyzer = mediapipe_analyzer.MediaPipeAnalyzer(
                running_mode=running_mode,
                face_result_callback=self.on_face_result if running_mode == vision.RunningMode.LIVE_STREAM else None,
                hand_result_callback=self.on_hand_result if running_mode == vision.RunningMode.LIVE_STREAM else None,
                enable_hand_detection=self.config_args.get('enable_mediapipe_hand', True),
                enable_distracted_detection=self.config_args.get('enable_mediapipe_distracted', True)
            )
        else: # MediaPipe가 비활성화된 경우 None으로 명시적 설정
            self.mediapipe_analyzer = None

        # Handle image sequence vs video/webcam
        if self.is_image_sequence:
            # print(f"[VideoThread] Processing image sequence with {len(self.image_files)} images")
            self.process_image_sequence()
        else:
            # 사용자 선택 회전 설정 가져오기
            user_rotation = self.config_args.get('video_rotation', 0)
            if user_rotation == 0:
                # 사용자가 "No Rotation" 선택한 경우에만 EXIF 정보 읽기
                if not source.isdigit() and Path(source).exists():
                    self.exif_orientation = get_exif_orientation(source)
                    print(f"[VideoThread] EXIF orientation: {self.exif_orientation}")
            else:
                # 사용자가 회전을 선택한 경우 해당 회전 적용
                rotation_map = {1: 6, 2: 8, 3: 3}  # 1=90°CW, 2=90°CCW, 3=180°
                self.exif_orientation = rotation_map.get(user_rotation, 1)
                print(f"[VideoThread] User selected rotation: {user_rotation} -> orientation: {self.exif_orientation}")
            
            # Original video/webcam processing
            self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print(f"Attempting with CAP_V4L2 failed. Retrying with default. Error: Could not open video source {source}")
                self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_GSTREAMER)
                if not self.cap.isOpened():
                    print(f"Attempting with CAP_GSTREAMER failed. Retrying with default. Error: Could not open video source {source}")
                    self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                    if not self.cap.isOpened():
                        print(f"Failed to open video source {source} with any specified backend.")
                        self.is_running = False
                        return

            prev_frame_time = 0
            while self.is_running and self.cap.isOpened():
                # --- FPS 제한 로직 ---
                current_time = time.time()
                if current_time - self.last_frame_time < self.frame_interval:
                    time.sleep(0.001)  # 1ms 대기
                    continue
                
                self.last_frame_time = current_time

                # --- 안전한 시킹 처리 ---
                if self.requested_seek_frame is not None:
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        target_frame = int((self.requested_seek_frame / 100.0) * total_frames)
                        target_frame = max(0, min(target_frame, total_frames - 1))
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        print(f"[VideoThread] Video position set to {self.requested_seek_frame}% (frame {target_frame}/{total_frames})")
                    self.requested_seek_frame = None

                ret, frame = self.cap.read()
                if not ret:
                    print("End of stream or cannot read frame.")
                    break

                im0 = frame.copy()
                
                # EXIF 회전 정보 적용
                im0 = apply_exif_rotation(im0, self.exif_orientation)
                
                frame_h, frame_w, _ = im0.shape # 현재 프레임의 크기

                    # Process frame with all analyzers
                self.process_frame(im0, frame_h, frame_w, enable_mediapipe, 
                                    enable_openvino, conf_thres, iou_thres, max_det, hide_labels, hide_conf)

            self.cap.release()
            # print("[VideoThread] Video capture released.")
        
        self.is_running = False
        if self.ros2_publisher:
            self.ros2_publisher.destroy_node()
        if ROS2_AVAILABLE and rclpy.ok():
            rclpy.shutdown()

    def process_image_sequence(self):
        """Process a sequence of images without stretching"""
        while self.is_running and self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to read image: {image_path}")
                self.current_image_index += 1
                continue
            
            im0 = frame.copy()
            frame_h, frame_w, _ = im0.shape
            
            # Process frame with all analyzers
            self.process_frame(im0, frame_h, frame_w, 
                             self.config_args.get('enable_mediapipe', False),
                             self.config_args.get('enable_openvino', False),
                             self.config_args.get('conf_thres', 0.25),
                             self.config_args.get('iou_thres', 0.45),
                             self.config_args.get('max_det', 10),
                             self.config_args.get('hide_labels', False),
                             self.config_args.get('hide_conf', False))
            
            # Move to next image after a delay
            time.sleep(2.0)  # 2 second delay between images
            self.current_image_index += 1

    def process_frame(self, im0, frame_h, frame_w, enable_mediapipe, 
                     enable_openvino, conf_thres, iou_thres, max_det, hide_labels, hide_conf):
        """Process a single frame with all enabled analyzers"""
        
        # --- 0. Crop 먼저 적용 ---
        im0 = self.apply_aspect_crop(im0)
        # Visualizer에 crop offset 설정은 더이상 필요 없음
        # 이하 기존 분석/시각화 코드 유지
        # --- 1. YOLOv5 Detection ---
        yolo_dets = torch.tensor([])
        if enable_mediapipe:
            mediapipe_results = {}
            if self.mediapipe_analyzer:
                # 선택된 모드에 따라 다른 함수 호출
                if self.mediapipe_analyzer.running_mode == vision.RunningMode.LIVE_STREAM:
                    timestamp = int(time.time() * 1000)
                    self.mediapipe_analyzer.detect_async(im0.copy(), timestamp)
                    # 비동기 모드에서는 콜백에서 업데이트된 최신 결과를 사용
                    mediapipe_results = self.mediapipe_analyzer._process_results(
                        self.latest_face_result, self.latest_hand_result
                    )
                else: # VIDEO 모드 (동기)
                    mediapipe_results = self.mediapipe_analyzer.analyze_frame(im0.copy())

                # --- MediaPipe 정면 캘리브레이션 트리거 처리 ---
                if self.mediapipe_calibration_trigger:
                    print(f"[VideoThread] MediaPipe calibration triggered. Face landmarks: {mediapipe_results.get('face_landmarks') is not None}")
                    if mediapipe_results.get("face_landmarks"):
                        calibrated_successfully = self.mediapipe_analyzer.calibrate_front_pose(
                            (frame_h, frame_w), mediapipe_results["face_landmarks"]
                        )
                        if calibrated_successfully:
                            print("[VideoThread] MediaPipe front pose calibrated successfully.")
                        else:
                            print("[VideoThread] MediaPipe front pose calibration failed.")
                    else:
                        print("[VideoThread] MediaPipe front pose calibration failed (no face detected).")
                    self.mediapipe_calibration_trigger = False # 캘리브레이션 요청 초기화

                # --- 4. Visualization ---
                if mediapipe_results:
                    im0 = self.visualizer_instance.draw_mediapipe_results(im0, mediapipe_results)
                    
                    # ROI 시각화 추가 (enable_face_position_filtering이 True일 때만)
                    if mediapipe_results and self.mediapipe_analyzer.enable_face_position_filtering:
                        roi_bounds = mediapipe_results.get("face_roi_bounds")
                        is_calibrated = mediapipe_results.get("mp_is_calibrated", False)
                        is_face_in_roi = mediapipe_results.get("is_driver_present", True)
                        im0 = self.visualizer_instance.draw_mediapipe_roi(im0, roi_bounds, is_calibrated, is_face_in_roi)
                    
                    # 디버깅: MediaPipe 결과 출력
                    if mediapipe_results:
                        # print(f"[DEBUG] MediaPipe results keys: {list(mediapipe_results.keys())}")
                        # if mediapipe_results.get("is_drowsy"):
                        #     print("[DEBUG] MediaPipe: Drowsy detected!")
                        # if mediapipe_results.get("is_yawning"):
                        #     print("[DEBUG] MediaPipe: Yawning detected!")
                        # if mediapipe_results.get("is_pupil_gaze_deviated"):
                        #     print("[DEBUG] MediaPipe: Pupil gaze deviated!")
                        # if mediapipe_results.get("is_dangerous_condition"):
                        #     print("[DEBUG] MediaPipe: Dangerous condition detected!")
                        pass
                    else:
                        # print("[DEBUG] MediaPipe results is empty")
                        pass

                # --- 5. Driver Status Analysis ---
                if mediapipe_results:
                    self.analyze_driver_status_mediapipe(mediapipe_results)

                # --- 6. Socket Communication ---
                    # YOLO 결과를 사람이 읽을 수 있는 dict 리스트로 변환
                yolo_results = []
                if mediapipe_results:
                    yolo_results.append({
                        'bbox': [], # No YOLO results in this mode
                        'conf': 0,
                        'class_id': -1,
                        'class_name': 'N/A'
                    })

                result_to_send = {
                    'yolo': yolo_results,
                    'mediapipe': mediapipe_results,
                    'openvino': {}, # OpenVINO 결과는 더 이상 사용되지 않으므로 빈 딕셔너리로 설정
                    'status': self.driver_status
                }
                    # print(f"result_to_send: {result_to_send}")
                # Only send if enabled and publisher exists
                if self.enable_ros2_sending and self.ros2_publisher:
                    self.ros2_publisher.publish_image(im0) # 이미지만 전송
                
                # --- FPS 계산 및 표시 ---
                if self.prev_frame_time is None:
                    self.prev_frame_time = time.time()
                new_frame_time = time.time()
                fps = 1/(new_frame_time - self.prev_frame_time) if (new_frame_time - self.prev_frame_time) > 0 else 0
                self.prev_frame_time = new_frame_time
                im0 = self.visualizer_instance.draw_fps(im0, fps)
                    
                # --- 진행률 계산 및 emit ---
                try:
                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        progress_percent = int(100 * current_frame / total_frames)
                    else:
                        progress_percent = 0
                except Exception:
                    progress_percent = 0
                self.change_pixmap_signal.emit(im0, progress_percent)
                    # time.sleep(0.01) # CPU 사용량 조절을 위해 필요시 주석 해제

    def stop(self):
        self.is_running = False
        # print("[VideoThread] Stopping video thread...")
        self.wait()
        if self.ros2_publisher:
            self.ros2_publisher.destroy_node()
        if ROS2_AVAILABLE and rclpy.ok():
            rclpy.shutdown()

    def analyze_driver_status(self, dlib_results):
        """Analyze driver status using Dlib results"""
        # This method can be implemented based on your existing driver status analysis logic
        pass

    def analyze_driver_status_mediapipe(self, mediapipe_results):
        """Analyze driver status using MediaPipe results"""
        # This method can be implemented based on your existing driver status analysis logic
        pass

    def apply_aspect_crop(self, frame):
        # aspect_ratio_mode: 0=Stretch, 1=Fit, 2=Crop, 3=Crop(Top)
        aspect_index = getattr(self, 'aspect_ratio_mode', 0)
        crop_offset = getattr(self, '_crop_offset', 0)
        label_width = 800  # image_label width (고정)
        label_height = 600 # image_label height (고정)
        h, w, ch = frame.shape
        # Stretch, Fit은 crop 없음
        if aspect_index == 0 or aspect_index == 1:
            return frame
        # Crop, Crop(Top)
        # 비율 유지하고 화면을 채우도록 조정 (잘림 발생 가능)
        import cv2
        from PyQt5.QtCore import Qt
        scaled = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        # 원본 비율 유지 crop
        frame_aspect = w / h
        label_aspect = label_width / label_height
        if frame_aspect > label_aspect:
            # 좌우가 더 넓음: 좌우를 crop
            new_w = int(h * label_aspect)
            x = (w - new_w) // 2
            cropped = frame[:, x:x+new_w]
        else:
            # 상하가 더 큼: 상하를 crop
            new_h = int(w / label_aspect)
            if aspect_index == 2:
                # 중앙 기준 crop + offset
                y = (h - new_h) // 2 + crop_offset
            else:
                # 상단 기준 crop + offset
                y = 0 + crop_offset
            y = max(0, min(y, h - new_h))
            cropped = frame[y:y+new_h, :]
        # 최종 리사이즈
        cropped = cv2.resize(cropped, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        return cropped

    def set_playback_fps(self, fps):
        """외부에서 재생 fps를 동적으로 변경"""
        self.target_fps = max(5.0, min(60.0, fps))
        self.frame_interval = 1.0 / self.target_fps

    def set_video_position(self, position_percent):
        """외부에서 비디오 위치를 동적으로 변경 (0-100%)"""
        self.requested_seek_frame = position_percent


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection with YOLOv5 & Dlib & MediaPipe")
        
        # GUI 상태 먼저 로드
        self.gui_state = self._load_gui_state()
        # source가 파일일 경우 존재하지 않으면 0(웹캠)으로 대체
        source = self.gui_state.get("source", "0")
        if source not in ["0", 0] and not Path(source).exists():
            print(f"[INFO] Saved source file '{source}' not found. Defaulting to webcam (0).")
            self.gui_state["source"] = "0"
        
        # 저장된 창 크기 복원, 없으면 기본값 사용
        window_geometry = self.gui_state.get("window_geometry", {"x": 100, "y": 100, "width": 1000, "height": 700})
        self.setGeometry(window_geometry["x"], window_geometry["y"], window_geometry["width"], window_geometry["height"])

        self.video_thread = None
        self.is_set_mediapipe_front_face_mode = False # MediaPipe 정면 설정 모드 상태
        self.is_set_openvino_front_face_mode = False
        self.mediapipe_analyzer = None
        self.openvino_analyzer = None
        self.openvino_hybrid_analyzer = None  # OpenVINO 하이브리드 분석기 추가
        
        # 소켓 전송용 IP/Port - 저장된 상태에서 로드
        # self.socket_ip = self.gui_state.get("socket_ip", "127.0.0.1")
        # self.socket_port = self.gui_state.get("socket_port", 5001)
        self.config_manager = ConfigManager()
        
        self.init_ui() # init_ui()를 먼저 호출하여 위젯을 생성합니다.

        self.thread = None
        self.is_running = False
        
        # 아이콘 설정 (실행 파일에서도 동작하도록)
        try:
            self.setWindowIcon(QIcon(os.path.join(ROOT, 'icons', 'icon.png')))
        except Exception as e:
            print(f"Error setting window icon: {e}")

        self.slider_position_user_changing = False  # 사용자가 슬라이더 조작 중인지 플래그
        # (신호 연결은 init_ui에서)

        if ROS2_AVAILABLE:
            rclpy.init(args=None)
            self.image_publisher = ImagePublisher()
        else:
            self.image_publisher = None

    def init_ui(self):
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        video_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 600)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.image_label.setScaledContents(True)
        video_layout.addWidget(self.image_label)
        
        self.image_label.mousePressEvent = self.image_label_mouse_press_event

        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("Stop Detection")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        self.btn_next_image = QPushButton("Next Image")
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_next_image.setEnabled(False)

        self.chk_mediapipe = QCheckBox("Enable MediaPipe")
        self.chk_mediapipe.setChecked(self.gui_state.get("enable_mediapipe", False))
        self.chk_mediapipe.stateChanged.connect(self.update_front_face_checkbox_states)

        # --- Hand Off 토글 버튼 추가 ---
        self.btn_hand_off = QPushButton("Hand On")
        self.btn_hand_off.setCheckable(True)
        # 버튼이 눌려있으면 hand detection ON, 아니면 OFF
        self.btn_hand_off.setChecked(self.gui_state.get("enable_mediapipe_hand", True))
        self.update_hand_off_button_style()
        self.btn_hand_off.toggled.connect(self.update_hand_off_button_style)

        # --- Distracted On 토글 버튼 추가 ---
        self.btn_distracted_off = QPushButton("Distracted On")
        self.btn_distracted_off.setCheckable(True)
        # 버튼이 눌려있으면 distracted detection ON, 아니면 OFF
        self.btn_distracted_off.setChecked(self.gui_state.get("enable_mediapipe_distracted", True))
        self.update_distracted_off_button_style()
        self.btn_distracted_off.toggled.connect(self.update_distracted_off_button_style)

        # 통합 캘리브레이션 버튼으로 변경
        self.btn_calibrate = QPushButton("Calibrate Front Face")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.clicked.connect(self.toggle_calibration_mode)
        self.is_calibration_mode = self.gui_state.get("is_calibration_mode", False)
        
        # 캘리브레이션 모드가 저장되어 있다면 버튼 상태 복원
        if self.is_calibration_mode:
            self.btn_calibrate.setText("Cancel Calibration")
            self.btn_calibrate.setStyleSheet("background-color: #ff6b6b; color: white;")

        # 설정 편집 버튼 추가
        self.btn_config = QPushButton("Edit Config")
        self.btn_config.clicked.connect(self.open_config_editor)

        # 설정 다시 로드 버튼 추가
        self.btn_reload_config = QPushButton("Reload Config")
        self.btn_reload_config.clicked.connect(self.reload_config)

        # 소켓 IP/Port 입력란 추가
        # socket_layout = QHBoxLayout()
        # self.label_socket_ip = QLabel("Socket IP:")
        # self.txt_socket_ip = QLineEdit(self.gui_state.get("socket_ip", "127.0.0.1"))
        # self.txt_socket_ip.textChanged.connect(self.save_socket_ip_port)
        # self.label_socket_port = QLabel("Port:")
        # self.txt_socket_port = QLineEdit(str(self.gui_state.get("socket_port", 5001)))
        # self.txt_socket_port.textChanged.connect(self.save_socket_ip_port)
        # socket_layout.addWidget(self.label_socket_ip)
        # socket_layout.addWidget(self.txt_socket_ip)
        # socket_layout.addWidget(self.label_socket_port)
        # socket_layout.addWidget(self.txt_socket_port)
        
        # --- Add socket send enable checkbox ---
        self.chk_send_ros2 = QCheckBox("ROS2로 데이터 전송")
        self.chk_send_ros2.setChecked(self.gui_state.get("enable_ros2_sending", True))
        # socket_layout.addWidget(self.chk_send_socket)

        # --- 비디오 회전 옵션 추가 ---
        rotation_layout = QHBoxLayout()
        self.label_rotation = QLabel("Video Rotation:")
        self.combo_rotation = QComboBox()
        self.combo_rotation.addItems(["No Rotation", "90° Clockwise", "90° Counter-clockwise", "180°"])
        # 저장된 회전 설정 로드
        rotation_index = self.gui_state.get("video_rotation", 0)
        self.combo_rotation.setCurrentIndex(rotation_index)
        rotation_layout.addWidget(self.label_rotation)
        rotation_layout.addWidget(self.combo_rotation)
        
        # --- Aspect Ratio 옵션 추가 ---
        self.label_aspect = QLabel("Aspect Ratio:")
        self.combo_aspect = QComboBox()
        self.combo_aspect.addItems(["Stretch", "Fit", "Crop", "Crop (Top)"])
        # 저장된 aspect ratio 설정 로드
        aspect_index = self.gui_state.get("aspect_ratio", 0)
        self.combo_aspect.setCurrentIndex(aspect_index)
        rotation_layout.addWidget(self.label_aspect)
        rotation_layout.addWidget(self.combo_aspect)
        
        # --- Crop Offset 옵션 추가 ---
        self.label_crop_offset = QLabel("Crop Offset:")
        self.txt_crop_offset = QLineEdit()
        self.txt_crop_offset.setFixedWidth(60)
        self.txt_crop_offset.setPlaceholderText("0")
        # 저장된 crop offset 설정 로드
        crop_offset = self.gui_state.get("crop_offset", "0")
        self.txt_crop_offset.setText(crop_offset)
        # 값 변경 시 VideoThread에 반영
        self.txt_crop_offset.textChanged.connect(self.update_crop_offset)
        rotation_layout.addWidget(self.label_crop_offset)
        rotation_layout.addWidget(self.txt_crop_offset)
        rotation_layout.addStretch()  # 우측 공간 확보
        rotation_layout.addWidget(self.btn_start)
        rotation_layout.addWidget(self.btn_stop)
        rotation_layout.addWidget(self.btn_next_image)
        
        # --- Video Speed Slider 추가 ---
        self.label_speed = QLabel("Playback Speed:")
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setMinimum(-100)
        self.slider_speed.setMaximum(100)
        # 저장된 playback speed 값 로드
        saved_speed = self.gui_state.get("playback_speed", 0)
        self.slider_speed.setValue(saved_speed)
        self.slider_speed.setTickInterval(10)
        self.slider_speed.setTickPosition(QSlider.TicksBelow)
        self.slider_speed.setFixedWidth(200)
        # 저장된 값에 따라 speed 표시 라벨 업데이트
        speed = 1.0 + (saved_speed / 100.0)
        speed = max(0.5, min(2.0, speed))
        self.label_speed_value = QLabel(f"{speed:.2f}x")
        self.slider_speed.valueChanged.connect(self.update_playback_speed)
        
        # --- Video Position Slider 추가 ---
        self.label_position = QLabel("Video Position:")
        self.slider_position = QSlider(Qt.Horizontal)
        self.slider_position.setMinimum(0)
        self.slider_position.setMaximum(100)
        self.slider_position.setValue(0)  # 0% = 시작 위치
        self.slider_position.setTickInterval(10)
        self.slider_position.setTickPosition(QSlider.TicksBelow)
        self.slider_position.setFixedWidth(200)
        self.label_position_value = QLabel("0%")
        self.slider_position.valueChanged.connect(self.update_video_position)
        # 슬라이더 동기화 신호 연결 (여기서 해야 linter 오류 없음)
        self.slider_position.sliderPressed.connect(self.on_slider_pressed)
        self.slider_position.sliderReleased.connect(self.on_slider_released)
        # 슬라이더들을 layout에 추가
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.label_speed)
        speed_layout.addWidget(self.slider_speed)
        speed_layout.addWidget(self.label_speed_value)
        speed_layout.addStretch()
        speed_layout.addWidget(self.label_position)
        speed_layout.addWidget(self.slider_position)
        speed_layout.addWidget(self.label_position_value)
        speed_layout.addStretch()
        # rotation_layout 아래에 추가
        main_layout.addLayout(speed_layout)

        self.update_front_face_checkbox_states() # 초기 상태 설정

        self.label_source = QLabel("Video Source (0 for webcam):")
        self.txt_source = DragDropLineEdit()
        self.txt_source.setText(self.gui_state.get("source", "0"))
        self.btn_browse_source = QPushButton("Browse")
        self.btn_browse_source.clicked.connect(self.browse_video_source)

        self.label_weights = QLabel("MediaPipe Weights:")
        self.txt_weights = QLineEdit(str(os.path.join(ROOT, 'weights', 'last.pt')))
        self.txt_weights.setText(self.gui_state.get("weights", str(os.path.join(ROOT, 'weights', 'last.pt'))))
        self.txt_weights.textChanged.connect(self.update_weights)
        self.btn_browse_weights = QPushButton("Browse")
        self.btn_browse_weights.clicked.connect(self.browse_weights)

        control_layout.addWidget(self.chk_mediapipe)
        control_layout.addWidget(self.btn_hand_off)
        control_layout.addWidget(self.btn_distracted_off)
        control_layout.addWidget(self.btn_calibrate)
        control_layout.addWidget(self.chk_send_ros2) # 추가
        control_layout.addStretch()

        source_weights_layout = QHBoxLayout()
        source_weights_layout.addWidget(self.label_source)
        source_weights_layout.addWidget(self.txt_source)
        source_weights_layout.addWidget(self.btn_browse_source)
        source_weights_layout.addWidget(self.label_weights)
        source_weights_layout.addWidget(self.txt_weights)
        source_weights_layout.addWidget(self.btn_browse_weights)
        source_weights_layout.addStretch()  # 우측 공간 확보
        source_weights_layout.addWidget(self.btn_config)
        source_weights_layout.addWidget(self.btn_reload_config)  # 추가

        main_layout.addLayout(source_weights_layout)
        # main_layout.addLayout(socket_layout) # 삭제된 코드
        main_layout.addLayout(rotation_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(video_layout)

        self.setLayout(main_layout)

    def browse_weights(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select MediaPipe Weights File", "", "PyTorch Weights (*.pt);;All Files (*)", options=options)
        if fileName:
            self.txt_weights.setText(fileName)

    def browse_video_source(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video or Image File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v);;Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif *.webp);;All Files (*)", 
            options=options
        )
        if fileName:
            self.txt_source.setText(fileName)

    def update_front_face_checkbox_states(self):
        """캘리브레이션 버튼 활성화 상태 업데이트"""
        # mediapipe, 또는 openvino 중 하나라도 활성화되어 있으면 캘리브레이션 버튼 활성화
        self.btn_calibrate.setEnabled(
            self.chk_mediapipe.isChecked()
        )
        # Hand Off 버튼 활성화/비활성화
        self.btn_hand_off.setEnabled(self.chk_mediapipe.isChecked())
        self.btn_distracted_off.setEnabled(self.chk_mediapipe.isChecked())
        # 캘리브레이션 모드가 활성화되어 있는데 해당 분석기가 비활성화되면 캘리브레이션 모드도 해제
        if self.is_calibration_mode:
            if not self.chk_mediapipe.isChecked():
                self.toggle_calibration_mode()  # 캘리브레이션 모드 해제

    def start_detection(self):
        # 항상 config를 최신으로 반영
        from config_manager import config_manager
        config_manager.reload()
        if self.thread is not None and self.thread.isRunning():
            return

        # 'Start' 버튼을 누르는 시점에 현재 GUI 상태를 읽어옵니다.
        config_args = {
            'source': self.txt_source.text(),
            'weights': self.txt_weights.text(),
            'enable_mediapipe': self.chk_mediapipe.isChecked(),
            'enable_mediapipe_hand': self.btn_hand_off.isChecked(),
            'enable_mediapipe_distracted': self.btn_distracted_off.isChecked(),
            'enable_ros2_sending': self.chk_send_ros2.isChecked(),
            'video_rotation': self.combo_rotation.currentIndex(),
            'aspect_ratio': self.combo_aspect.currentIndex(),
            'crop_offset': self.txt_crop_offset.text(),
            'playback_speed': self.slider_speed.value()
        }

        self.thread = VideoThread(config_args)
        self.thread.change_pixmap_signal.connect(self.update_image)
        # VideoThread에 현재 정면 설정 모드 상태 전달 (새로운 속성 사용)
        self.thread.set_mediapipe_front_face_mode = self.is_set_mediapipe_front_face_mode

        # --- 재생 속도 슬라이더 값 반영 ---
        v = self.slider_speed.value()
        speed = 1.0 + (v / 100.0)
        speed = max(0.5, min(2.0, speed))
        base_fps = 20.0
        new_fps = base_fps * speed
        new_fps = max(5.0, min(60.0, new_fps))
        self.thread.set_playback_fps(new_fps)
        self.label_speed_value.setText(f"{speed:.2f}x")

        self.thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_next_image.setEnabled(self.thread.is_image_sequence)

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.clear()
        
        # 감지 중지 시 캘리브레이션 모드 비활성화
        if self.is_calibration_mode:
            self.is_calibration_mode = False
            self.btn_calibrate.setText("Calibrate Front Face")
            self.btn_calibrate.setStyleSheet("")
            print("Calibration mode automatically disabled due to detection stop.")
        
        # 감지 중지 시에는 정면 설정 체크박스를 다시 활성화
        self.update_front_face_checkbox_states()
        self.is_set_mediapipe_front_face_mode = False

    def on_slider_pressed(self):
        self.slider_position_user_changing = True
    def on_slider_released(self):
        self.slider_position_user_changing = False
        # 슬라이더 놓을 때 VideoThread에 위치 변경 요청
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            position_percent = self.slider_position.value()
            self.thread.set_video_position(position_percent)
            self.label_position_value.setText(f"{position_percent}%")

    @pyqtSlot(np.ndarray, int)
    def update_image(self, cv_img, progress_percent):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        self.image_label.setPixmap(pixmap)
        # --- 슬라이더 자동 갱신 ---
        if not self.slider_position_user_changing:
            self.slider_position.setValue(progress_percent)
            self.label_position_value.setText(f"{progress_percent}%")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(scaled_img)

    def toggle_calibration_mode(self):
        """통합 캘리브레이션 모드 토글"""
        self.is_calibration_mode = not self.is_calibration_mode
        
        if self.is_calibration_mode:
            # 캘리브레이션 모드 활성화
            self.btn_calibrate.setText("Cancel Calibration")
            self.btn_calibrate.setStyleSheet("background-color: #ff6b6b; color: white;")
            
            # 활성화된 분석기에 따라 캘리브레이션 모드 설정
            if self.chk_mediapipe.isChecked():
                self.is_set_mediapipe_front_face_mode = True
                if self.thread:
                    self.thread.set_mediapipe_front_face_mode = True
                    
        else:
            # 캘리브레이션 모드 비활성화
            self.btn_calibrate.setText("Calibrate Front Face")
            self.btn_calibrate.setStyleSheet("")
            
            # 모든 캘리브레이션 모드 해제
            self.is_set_mediapipe_front_face_mode = False
            
            if self.thread:
                self.thread.set_mediapipe_front_face_mode = False

    def image_label_mouse_press_event(self, ev: 'QMouseEvent'):
        """통합 캘리브레이션 모드에서 마우스 클릭 처리"""
        if not self.is_calibration_mode or not self.thread or not self.thread.isRunning():
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            print("Mouse clicked to trigger calibration.")
            # 활성화된 분석기에 따라 캘리브레이션 트리거 설정
            if self.chk_mediapipe.isChecked() and self.is_set_mediapipe_front_face_mode:
                self.thread.mediapipe_calibration_trigger = True
                print("MediaPipe calibration triggered.")

    def open_config_editor(self):
        """설정 편집기를 엽니다."""
        try:
            from config_manager import config_manager
            
            # 현재 설정을 표시
            config_manager.print_config()
            
            # 설정 파일을 텍스트 에디터로 열기
            import subprocess
            import platform
            import os
            
            config_file_path = config_manager.config_file.absolute()
            
            if platform.system() == "Windows":
                subprocess.run(["notepad", str(config_file_path)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-t", str(config_file_path)])
            else:  # Linux
                # Linux에서 텍스트 에디터 우선순위로 시도
                editors = ["gedit", "nano", "vim", "mousepad", "leafpad", "kate", "geany"]
                
                for editor in editors:
                    try:
                        # 에디터가 설치되어 있는지 확인
                        result = subprocess.run(["which", editor], capture_output=True, text=True)
                        if result.returncode == 0:
                            subprocess.run([editor, str(config_file_path)])
                            break
                    except:
                        continue
                else:
                    # 모든 에디터가 실패하면 xdg-open 사용
                    subprocess.run(["xdg-open", str(config_file_path)])
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"설정 편집기를 열 수 없습니다: {e}")

    def reload_config(self):
        """설정 파일을 다시 로드하고 실행 중인 감지 스레드를 재시작합니다."""
        try:
            from config_manager import config_manager
            config_manager.reload()
            
            # 실행 중인 감지 스레드가 있으면 재시작
            if self.thread and self.thread.isRunning():
                print("Restarting detection thread to apply new configuration...")
                self.stop_detection()
                import time
                time.sleep(0.5)  # 잠시 대기
                self.start_detection()
                QMessageBox.information(self, "Config Reloaded", 
                                      "Configuration reloaded and detection restarted successfully.")
            else:
                print("Configuration reloaded successfully.")
                QMessageBox.information(self, "Config Reloaded", "Configuration reloaded successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to reload configuration: {e}")

    def closeEvent(self, event):
        """
        PyQt에서 창이 닫힐 때 자동으로 호출되는 함수입니다.
        """
        print("Closing application...")
        self._save_gui_state()
        self.stop_detection()
        event.accept()         # 창 닫기 허용

    def next_image(self):
        """Manually advance to the next image in the sequence"""
        if self.thread and hasattr(self.thread, 'is_image_sequence') and self.thread.is_image_sequence:
            if self.thread.current_image_index < len(self.thread.image_files) - 1:
                self.thread.current_image_index += 1
                # Force the thread to process the next image
                self.thread.process_image_sequence()
            else:
                QMessageBox.information(self, "End of Sequence", "This is the last image in the sequence.")

    def _load_gui_state(self):
        try:
            with open(GUI_STATE_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {} # Return empty dict if file not found or invalid

    def _save_gui_state(self):
        # 현재 창의 위치와 크기 가져오기
        window_geometry = {
            "x": self.geometry().x(),
            "y": self.geometry().y(),
            "width": self.geometry().width(),
            "height": self.geometry().height()
        }
        
        state = {
            "window_geometry": window_geometry,
            "enable_mediapipe": self.chk_mediapipe.isChecked(),
            "enable_mediapipe_hand": self.btn_hand_off.isChecked(),
            "enable_mediapipe_distracted": self.btn_distracted_off.isChecked(),
            "source": self.txt_source.text(),
            "weights": self.txt_weights.text(),
            "enable_ros2_sending": self.chk_send_ros2.isChecked(),
            "is_set_mediapipe_front_face_mode": self.is_set_mediapipe_front_face_mode,
            "is_calibration_mode": False,  # 항상 OFF로 저장
            "video_rotation": self.combo_rotation.currentIndex(),
            "aspect_ratio": self.combo_aspect.currentIndex(),
            "crop_offset": self.txt_crop_offset.text(),
            "playback_speed": self.slider_speed.value()
        }
        try:
            with open(GUI_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"Error saving GUI state: {e}")

    def update_hand_off_button_style(self):
        if self.btn_hand_off.isChecked():
            # ON 상태(눌림): 초록색
            self.btn_hand_off.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            # OFF 상태(해제): 기본
            self.btn_hand_off.setStyleSheet("")

    def update_distracted_off_button_style(self):
        if self.btn_distracted_off.isChecked():
            # ON 상태(눌림): 초록색
            self.btn_distracted_off.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            # OFF 상태(해제): 기본
            self.btn_distracted_off.setStyleSheet("")

    def update_crop_offset(self):
        """Crop offset 값이 변경될 때 VideoThread에 반영"""
        try:
            if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
                crop_offset = int(self.txt_crop_offset.text())
                self.thread.crop_offset = crop_offset
        except ValueError:
            # 숫자가 아닌 값이 입력된 경우 무시
            pass

    def update_weights(self):
        """Weights 값이 변경될 때 VideoThread에 반영"""
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            self.thread.weights = self.txt_weights.text()

    def update_playback_speed(self):
        """슬라이더 값이 바뀔 때 VideoThread의 재생 속도(fps) 반영"""
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            v = self.slider_speed.value()
            # -100~+100 -> 0.5x~2.0x (0=1.0x)
            speed = 1.0 + (v / 100.0)
            speed = max(0.5, min(2.0, speed))
            # 기본 fps는 20, 0.5x=10fps, 2.0x=40fps
            base_fps = 20.0
            new_fps = base_fps * speed
            new_fps = max(5.0, min(60.0, new_fps))
            self.thread.set_playback_fps(new_fps)
            self.label_speed_value.setText(f"{speed:.2f}x")
        else:
            v = self.slider_speed.value()
            speed = 1.0 + (v / 100.0)
            speed = max(0.5, min(2.0, speed))
            self.label_speed_value.setText(f"{speed:.2f}x")

    def update_video_position(self):
        # 사용자가 슬라이더를 움직일 때만 호출됨
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            if self.slider_position_user_changing:
                position_percent = self.slider_position.value()
                self.label_position_value.setText(f"{position_percent}%")
        else:
            position_percent = self.slider_position.value()
            self.label_position_value.setText(f"{position_percent}%")


if __name__ == "__main__":
    # GTK/Gdk 경고 메시지 숨기기
    import os
    import sys
    
    # [Fix] Qt platform plugin conflict between PyQt5 and opencv-python.
    # This ensures that the application uses the Qt plugins bundled with PyQt5.
    from PyQt5.QtCore import QLibraryInfo
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
