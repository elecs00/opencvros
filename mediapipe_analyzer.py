# sleep/mediapipe_analyzer.py

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config_manager import get_mediapipe_config
from detector_utils import (
    get_mediapipe_head_pose_from_landmarks,
    get_mediapipe_head_pose_from_matrix,
    get_mediapipe_true_pitch_from_landmarks,
    get_mediapipe_pupil_center,
    calculate_mediapipe_pupil_gaze_deviation,
    calculate_mediapipe_face_center_and_size,
    is_mediapipe_face_within_calibrated_bounds
)

# Constants - now loaded from config
EYE_BLINK_THRESHOLD = get_mediapipe_config("eye_blink_threshold", 0.3)
EYE_BLINK_THRESHOLD_HEAD_UP = get_mediapipe_config("eye_blink_threshold_head_up", 0.2)
EYE_BLINK_THRESHOLD_HEAD_DOWN = get_mediapipe_config("eye_blink_threshold_head_down", 0.25)
HEAD_UP_THRESHOLD_FOR_EYE = get_mediapipe_config("head_up_threshold_for_eye", -10.0)
HEAD_DOWN_THRESHOLD_FOR_EYE = get_mediapipe_config("head_down_threshold_for_eye", 8.0)
JAW_OPEN_THRESHOLD = get_mediapipe_config("jaw_open_threshold", 0.4)
DROWSY_CONSEC_FRAMES = get_mediapipe_config("drowsy_consec_frames", 15)
YAWN_CONSEC_FRAMES = get_mediapipe_config("yawn_consec_frames", 30)
PITCH_DOWN_THRESHOLD = get_mediapipe_config("pitch_down_threshold", 10)
PITCH_UP_THRESHOLD = get_mediapipe_config("pitch_up_threshold", -15)
POSE_CONSEC_FRAMES = get_mediapipe_config("pose_consec_frames", 20)
GAZE_VECTOR_THRESHOLD = get_mediapipe_config("gaze_vector_threshold", 0.5)

# GUI 호환성을 위한 추가 상수들 - now loaded from config
MP_YAW_THRESHOLD = get_mediapipe_config("mp_yaw_threshold", 30.0)
MP_PITCH_THRESHOLD = get_mediapipe_config("mp_pitch_threshold", 10.0)
MP_ROLL_THRESHOLD = get_mediapipe_config("mp_roll_threshold", 999.0)
GAZE_THRESHOLD = get_mediapipe_config("gaze_threshold", 0.5)
DISTRACTION_CONSEC_FRAMES = get_mediapipe_config("distraction_consec_frames", 10)

# Hand detection sensitivity
MIN_HAND_DETECTION_CONFIDENCE = get_mediapipe_config("min_hand_detection_confidence", 0.3)
MIN_HAND_PRESENCE_CONFIDENCE = get_mediapipe_config("min_hand_presence_confidence", 0.3)
HAND_OFF_CONSEC_FRAMES = get_mediapipe_config("hand_off_consec_frames", 5)

TRUE_PITCH_THRESHOLD = get_mediapipe_config("true_pitch_threshold", 10.0)
HEAD_ROTATION_THRESHOLD_FOR_GAZE = get_mediapipe_config("head_rotation_threshold_for_gaze", 15.0)

# Hand size filtering constants
HAND_SIZE_RATIO_THRESHOLD = get_mediapipe_config("hand_size_ratio_threshold", 0.67)
ENABLE_HAND_SIZE_FILTERING = get_mediapipe_config("enable_hand_size_filtering", True)

# Pupil-based gaze detection constants
ENABLE_PUPIL_GAZE_DETECTION = get_mediapipe_config("enable_pupil_gaze_detection", True)
PUPIL_GAZE_THRESHOLD = get_mediapipe_config("pupil_gaze_threshold", 0.05)  # 얼굴 크기 대비 임계값
PUPIL_GAZE_CONSEC_FRAMES = get_mediapipe_config("pupil_gaze_consec_frames", 10)  # 연속 프레임 수

# config.json에서 프레임 임계값 읽기
WAKEUP_FRAME_THRESHOLD = get_mediapipe_config("wakeup_frame_threshold", 60)
DISTRACTED_FRAME_THRESHOLD = get_mediapipe_config("distracted_frame_threshold", 60)

class MediaPipeAnalyzer:
    def __init__(self, running_mode=None, face_result_callback=None, hand_result_callback=None, enable_hand_detection=True, enable_distracted_detection=True):
        # GUI 호환성을 위해 기본값 설정
        if running_mode is None:
            running_mode = vision.RunningMode.VIDEO
        
        self.running_mode = running_mode
        self.face_result_callback = face_result_callback
        self.hand_result_callback = hand_result_callback
        self.enable_hand_detection = enable_hand_detection
        self.enable_distracted_detection = enable_distracted_detection

        # 캘리브레이션 관련 변수들 (GUI 호환성)
        self.mp_front_face_offset_yaw = 0.0
        self.mp_front_face_offset_pitch = 0.0
        self.mp_front_face_offset_roll = 0.0
        self.calibrated_gaze_x = 0.0
        self.calibrated_gaze_y = 0.0
        self.gaze_deviated_frame_count = 0
        self.is_calibrated = False
        self.just_calibrated = False  # 방금 캘리브레이션 완료 플래그
        self.eyes_confirmed_closed = False  # Track if eyes are confirmed closed (like Dlib)
        
        # --- Face Position Calibration Variables ---
        self.calibrated_face_center = None  # (x, y) coordinates of calibrated face center
        self.calibrated_face_size = None    # (width, height) of calibrated face
        self.calibrated_face_roi = None     # (x1, y1, x2, y2) ROI bounds
        self.face_position_threshold = get_mediapipe_config("face_position_threshold", 0.3)  # Maximum allowed deviation from calibrated position (as ratio of face size)
        self.face_size_threshold = get_mediapipe_config("face_size_threshold", 0.5)      # Maximum allowed size difference (as ratio)
        self.enable_face_position_filtering = get_mediapipe_config("enable_face_position_filtering", True)  # Enable/disable face position filtering
        self.face_roi_scale = get_mediapipe_config("face_roi_scale", 1.5)  # Scale factor for face ROI detection area

        # --- Pupil-based Gaze Detection Variables ---
        self.calibrated_pupil_center = None  # (x, y) coordinates of calibrated pupil center
        self.pupil_gaze_deviated_frame_count = 0  # 연속 프레임 카운터
        self.enable_pupil_gaze_detection = ENABLE_PUPIL_GAZE_DETECTION
        self.pupil_gaze_threshold = PUPIL_GAZE_THRESHOLD
        self.pupil_gaze_consec_frames = PUPIL_GAZE_CONSEC_FRAMES

        models_dir = Path(__file__).parent
        face_model_path = models_dir / "models" / "face_landmarker.task"
        hand_model_path = models_dir / "models" / "hand_landmarker.task"

        # 모델 파일이 없으면 기본 MediaPipe Face Mesh 사용
        if not face_model_path.exists():
            print("[MediaPipeAnalyzer] Task API model not found, using traditional MediaPipe Face Mesh")
            self.use_task_api = False
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_landmarker = None
        else:
            self.use_task_api = True
            for model_path in [face_model_path, hand_model_path]:
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}.")

            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(face_model_path)),
                running_mode=running_mode,
                result_callback=self.face_result_callback
                if running_mode == vision.RunningMode.LIVE_STREAM
                else None,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

            if self.enable_hand_detection:
                hand_options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
                    running_mode=running_mode,
                    result_callback=self.hand_result_callback
                    if running_mode == vision.RunningMode.LIVE_STREAM
                    else None,
                    num_hands=2,
                    min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
                    min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            else:
                self.hand_landmarker = None

        self.is_closed = False
        (
            self.eye_closed_frame_count,
            self.yawn_frame_count,
            self.head_down_frame_count,
        ) = 0, 0, 0
        self.head_up_frame_count, self.gaze_deviated_frame_count = 0, 0
        self.left_hand_off_frame_count, self.right_hand_off_frame_count = 0, 0
        self.distraction_frame_counter = 0  # Add distraction frame counter for consecutive detection
        self.drowsy_frame_count = 0
        self.distracted_frame_count = 0
        self.driver_absent_frame_count = 0  # 운전자 미검출 연속 프레임 카운터
        self.driver_absent_frame_threshold = 30  # 30프레임(약 1.5초) 연속 미검출 시 운전자 없음 표시
        
        # 손 감지 연속 프레임 카운터 추가
        self.hand_detected_frame_count = 0  # 손 감지 연속 프레임 카운터
        self.hand_warning_frame_threshold = 60  # 60프레임 연속 손 감지 시 경고
        # print(f"[MediaPipeAnalyzer] Initialized in {running_mode.name} mode.")

    def close(self):
        if self.is_closed:
            return
        if self.use_task_api:
            self.face_landmarker.close()
            self.hand_landmarker.close()
        else:
            self.face_mesh.close()
        self.is_closed = True
        print("[MediaPipeAnalyzer] Landmarkers closed.")

    def calibrate_front_pose(self, frame_size, face_landmarks):
        """GUI 호환성을 위한 캘리브레이션 메서드"""
        if not face_landmarks:
            print("캘리브레이션: 랜드마크가 감지되지 않았습니다.")
            return False

        if self.use_task_api:
            # Task API 방식: 캘리브레이션 플래그만 설정하고 다음 프레임에서 오프셋 설정
            print("[MediaPipeAnalyzer] Task API calibration - will set offsets in next frame...")
            
            # 캘리브레이션 플래그 설정
            self.just_calibrated = True  # 다음 프레임에서 오프셋 설정
            self.is_calibrated = True
            
            # --- Face Position and Size Calibration for Task API ---
            face_info = calculate_mediapipe_face_center_and_size(face_landmarks, frame_size)
            if face_info:
                self.calibrated_face_center, self.calibrated_face_size, self.calibrated_face_roi = face_info
                print(f"[MediaPipeAnalyzer] Face position calibrated: center={self.calibrated_face_center}, "
                      f"size={self.calibrated_face_size}")
            
            # --- Pupil Position Calibration for Task API ---
            pupil_center = get_mediapipe_pupil_center(face_landmarks)
            if pupil_center is not None:
                self.calibrated_pupil_center = pupil_center
                print(f"[MediaPipeAnalyzer] Pupil position calibrated: center=({pupil_center[0]:.3f}, {pupil_center[1]:.3f})")
            else:
                print("[MediaPipeAnalyzer] Pupil position calibration failed")
            
            print("[MediaPipeAnalyzer] Task API front pose calibration flag set.")
            return True
        else:
            # 기존 Face Mesh 방식과 호환되도록 head pose 계산
            head_pose_data = get_mediapipe_head_pose_from_landmarks(frame_size, face_landmarks)
            
            if head_pose_data:
                self.mp_front_face_offset_yaw = -head_pose_data["yaw"]
                self.mp_front_face_offset_pitch = -head_pose_data["pitch"]
                self.mp_front_face_offset_roll = -head_pose_data["roll"]
                
                # --- Face Position and Size Calibration for Face Mesh ---
                face_info = calculate_mediapipe_face_center_and_size(face_landmarks, frame_size)
                if face_info:
                    self.calibrated_face_center, self.calibrated_face_size, self.calibrated_face_roi = face_info
                    print(f"[MediaPipeAnalyzer] Face position calibrated: center={self.calibrated_face_center}, "
                          f"size={self.calibrated_face_size}")
                
                # --- Pupil Position Calibration for Face Mesh ---
                pupil_center = get_mediapipe_pupil_center(face_landmarks)
                if pupil_center is not None:
                    self.calibrated_pupil_center = pupil_center
                    print(f"[MediaPipeAnalyzer] Pupil position calibrated: center=({pupil_center[0]:.3f}, {pupil_center[1]:.3f})")
                else:
                    print("[MediaPipeAnalyzer] Pupil position calibration failed")
                
                # gaze 캘리브레이션 - 실제 현재 gaze 값 계산
                try:
                    # 눈동자 위치를 기반으로 gaze 계산
                    left_eye_center = np.array([
                        (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2,
                        (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2
                    ])
                    right_eye_center = np.array([
                        (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2,
                        (face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2
                    ])
                    
                    # 얼굴 중심에서의 상대적 위치
                    face_center = np.array([0.5, 0.5])  # 정규화된 좌표
                    gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2 - face_center[0]
                    gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2 - face_center[1]
                    
                    self.calibrated_gaze_x = gaze_x
                    self.calibrated_gaze_y = gaze_y
                    
                    print(f"[MediaPipeAnalyzer] Gaze calibrated: x={gaze_x:.3f}, y={gaze_y:.3f}")
                except Exception as e:
                    print(f"[MediaPipeAnalyzer] Gaze calibration error: {e}")
                    self.calibrated_gaze_x = 0.0
                    self.calibrated_gaze_y = 0.0
                
                self.is_calibrated = True
                self.just_calibrated = True  # 방금 캘리브레이션 완료 표시
                print("[MediaPipeAnalyzer] Front pose calibrated successfully.")
                return True
            else:
                print("[MediaPipeAnalyzer] Front pose calibration failed.")
                return False



    def analyze_frame(self, frame):
        """GUI 호환성을 위한 analyze_frame 메서드"""
        if self.use_task_api:
            return self.detect_sync(frame)
        else:
            return self._analyze_frame_traditional(frame)

    def _analyze_frame_traditional(self, frame):
        """기존 MediaPipe Face Mesh 방식"""
        results = {
            "face_landmarks": None,
            "is_drowsy": False,
            "is_yawning": False,
            "is_distracted_from_front": False,
            "gaze_x": 0.0,
            "gaze_y": 0.0,
            "is_gaze": False,
            "mp_head_pitch_deg": 0.0,
            "mp_head_yaw_deg": 0.0,
            "mp_head_roll_deg": 0.0,
            "mp_is_calibrated": self.is_calibrated,
            "mp_is_distracted_from_front": False,
            "is_pupil_gaze_deviated": False,
            "enable_pupil_gaze_detection": self.enable_pupil_gaze_detection,
            "mp_head_pose_color": (100, 100, 100),  # Default grey color
            "enable_distracted_detection": self.enable_distracted_detection,
            # 졸음 해제 관련 필드들
            "wakeup_frame_threshold": WAKEUP_FRAME_THRESHOLD,
            "drowsy_frame_count": self.drowsy_frame_count
        }
        
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh_results = self.face_mesh.process(rgb_frame)
        
        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            frame_size = (frame.shape[0], frame.shape[1])
            
            # ROI 범위 체크 (enable_face_position_filtering이 활성화된 경우에만)
            if self.enable_face_position_filtering:
                if not is_mediapipe_face_within_calibrated_bounds(
                    face_landmarks, frame_size, self.calibrated_face_center, 
                    self.calibrated_face_size, self.face_roi_scale
                ):
                    self.driver_absent_frame_count += 1
                    if self.driver_absent_frame_count >= self.driver_absent_frame_threshold:
                        results["is_driver_present"] = False
                    else:
                        results["is_driver_present"] = True
                    results["is_distracted_no_face"] = True
                    # ROI 범위를 벗어난 얼굴은 랜드마크를 None으로 설정
                    results["face_landmarks"] = None
                    return results
            
            # ROI 필터링이 비활성화되었거나 ROI 범위 내의 얼굴 처리
            self.driver_absent_frame_count = 0
            results["is_driver_present"] = True
            results["face_landmarks"] = face_landmarks
            
            # --- Pupil-based Gaze Detection ---
            pupil_gaze_deviated = calculate_mediapipe_pupil_gaze_deviation(
                face_landmarks, frame_size, self.calibrated_pupil_center,
                self.pupil_gaze_threshold, self.pupil_gaze_consec_frames
            )
            results["is_pupil_gaze_deviated"] = pupil_gaze_deviated
            
            # Head pose 계산
            head_pose_data = get_mediapipe_head_pose_from_landmarks(frame_size, face_landmarks)
            
            if head_pose_data:
                # 캘리브레이션된 오프셋 적용
                current_pitch = head_pose_data["pitch"] + self.mp_front_face_offset_pitch
                current_yaw = head_pose_data["yaw"] + self.mp_front_face_offset_yaw
                current_roll = head_pose_data["roll"] + self.mp_front_face_offset_roll
                
                # 디버깅 출력 (캘리브레이션 상태 확인)
                # print(f"[DEBUG] Head pose: raw(pitch={head_pose_data['pitch']:.1f}, yaw={head_pose_data['yaw']:.1f}, roll={head_pose_data['roll']:.1f})")
                # print(f"[DEBUG] Head pose: offsets(pitch={self.mp_front_face_offset_pitch:.1f}, yaw={self.mp_front_face_offset_yaw:.1f}, roll={self.mp_front_face_offset_roll:.1f})")
                # print(f"[DEBUG] Head pose: calibrated(pitch={current_pitch:.1f}, yaw={current_yaw:.1f}, roll={current_roll:.1f})")
                
                results["mp_head_pitch_deg"] = current_pitch
                results["mp_head_yaw_deg"] = current_yaw
                results["mp_head_roll_deg"] = current_roll
                
                # Head pose color based on calibration status
                if not self.is_calibrated:
                    results["mp_head_pose_color"] = (100, 100, 100)  # Grey if not calibrated
                    # 캘리브레이션 전에는 값을 0으로 설정
                    results["mp_head_pitch_deg"] = 0.0
                    results["mp_head_yaw_deg"] = 0.0
                    results["mp_head_roll_deg"] = 0.0
                    results["is_head_down"] = False
                    results["is_head_up"] = False
                    results["is_distracted_from_front"] = False
                    results["mp_is_distracted_from_front"] = False
                else:
                    # Distracted detection이 비활성화된 경우 검출하지 않음
                    if not self.enable_distracted_detection:
                        results["mp_head_pose_color"] = (100, 100, 100)  # Grey when disabled
                        results["is_distracted_from_front"] = False
                        results["mp_is_distracted_from_front"] = False
                        results["is_look_ahead_warning"] = False
                    else:
                        # 정면 이탈 감지
                        if (abs(current_yaw) > MP_YAW_THRESHOLD or
                            abs(current_pitch) > MP_PITCH_THRESHOLD):
                            results["is_distracted_from_front"] = True
                            results["mp_is_distracted_from_front"] = True
                            results["mp_head_pose_color"] = (0, 0, 255)  # Red for distracted
                        else:
                            results["mp_head_pose_color"] = (0, 255, 0)  # Green for normal
                    
                    # 고개 숙임/들기 감지
                    if current_pitch > PITCH_DOWN_THRESHOLD:
                        results["is_head_down"] = True
                        results["is_head_up"] = False
                    elif current_pitch < PITCH_UP_THRESHOLD:
                        results["is_head_up"] = True
                        results["is_head_down"] = False
                    else:
                        results["is_head_down"] = False
                        results["is_head_up"] = False
                
                # Gaze 계산 (간단한 방식)
                # 눈동자 위치를 기반으로 gaze 계산
                left_eye_center = np.array([
                    (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2,
                    (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2
                ])
                right_eye_center = np.array([
                    (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2,
                    (face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2
                ])
                
                # 얼굴 중심에서의 상대적 위치
                face_center = np.array([0.5, 0.5])  # 정규화된 좌표
                gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2 - face_center[0]
                gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2 - face_center[1]
                
                results["gaze_x"] = gaze_x
                results["gaze_y"] = gaze_y
                
                # 방금 캘리브레이션 완료된 경우 현재 gaze 값을 캘리브레이션 값으로 업데이트
                if self.just_calibrated:
                    self.calibrated_gaze_x = gaze_x
                    self.calibrated_gaze_y = gaze_y
                    print(f"[MediaPipeAnalyzer] Gaze updated after calibration: x={gaze_x:.3f}, y={gaze_y:.3f}")
                # 자동 캘리브레이션 제거 - 사용자가 Calibrate 버튼을 클릭할 때만 캘리브레이션
                # elif not self.is_calibrated:
                #     self.calibrated_gaze_x = gaze_x
                #     self.calibrated_gaze_y = gaze_y
                #     self.is_calibrated = True  # 캘리브레이션 완료 표시
                #     print(f"[MediaPipeAnalyzer] Gaze auto-calibrated: x={gaze_x:.3f}, y={gaze_y:.3f}")
                
                # 시선 이탈 감지 (캘리브레이션 적용)
                gaze_diff = abs(gaze_x - self.calibrated_gaze_x)
                if gaze_diff > GAZE_THRESHOLD:
                    results["is_gaze"] = True
                else:
                    pass
                
                # 졸음/하품 감지 (간단한 방식)
                # 눈 크기로 졸음 감지
                left_eye_height = abs(face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
                right_eye_height = abs(face_landmarks.landmark[386].y - face_landmarks.landmark[374].y)
                avg_eye_height = (left_eye_height + right_eye_height) / 2
                
                if avg_eye_height < 0.02:  # 임계값 조정 필요
                    results["is_drowsy"] = True
                
                # 입 크기로 하품 감지
                mouth_height = abs(face_landmarks.landmark[13].y - face_landmarks.landmark[14].y)
                if mouth_height > 0.05:  # 임계값 조정 필요
                    results["is_yawning"] = True
        
        return results



    def _process_results(self, face_result, hand_result):
        results = {
            "face_landmarks": [],
            "pose_landmarks": [],
            "right_hand_landmarks": [],
            "left_hand_landmarks": [],
            "head_pose": None,
            "is_drowsy": False,
            "is_yawning": False,
            "is_head_down": False,
            "is_head_up": False,
            "is_gaze_deviated": False,
            "is_left_hand_off": False,
            "is_right_hand_off": False,
            "is_distracted_no_face": False,
            "ear_value": 0.0,
            "mar_value": 0.0,
            "gaze_vector": (0, 0),
            "all_blendshapes": {},
            # GUI 호환성을 위한 추가 필드들
            "mp_head_pitch_deg": 0.0,
            "mp_head_yaw_deg": 0.0,
            "mp_head_roll_deg": 0.0,
            "mp_is_calibrated": self.is_calibrated,
            "mp_is_distracted_from_front": False,
            "mp_head_pose_color": (100, 100, 100),  # Default grey color
            "gaze_x": 0.0,
            "gaze_y": 0.0,
            "is_gaze": False,
            "is_distracted_from_front": False,
            # 손 감지 관련 새로운 필드들
            "hand_status": "No Hands Detected",  # "No Hands", "One Hand", "Two Hands"
            "hand_warning_color": "green",  # "green", "yellow", "red"
            "hand_warning_message": "",
            # 위험 상태 감지 필드들
            "is_dangerous_condition": False,  # 눈 감음 + 고개 숙임
            "dangerous_condition_message": "",
            # 눈동자 기반 시선 감지 필드들
            "is_pupil_gaze_deviated": False,
            "pupil_gaze_deviation": 0.0,
            "enable_pupil_gaze_detection": self.enable_pupil_gaze_detection,
            "enable_distracted_detection": self.enable_distracted_detection,
            # 졸음 해제 관련 필드들
            "wakeup_frame_threshold": WAKEUP_FRAME_THRESHOLD,
            "drowsy_frame_count": self.drowsy_frame_count
        }
        
        if face_result and face_result.face_landmarks:
            # 실제 프레임 크기를 사용하도록 수정 (detect_sync에서 전달받은 프레임 크기 사용)
            frame_size = getattr(self, 'current_frame_size', (640, 480))  # 기본값은 640x480
            
            # ROI 범위 체크 (enable_face_position_filtering이 활성화된 경우에만)
            if self.enable_face_position_filtering and not self.just_calibrated:
                if not is_mediapipe_face_within_calibrated_bounds(
                    face_result.face_landmarks[0], frame_size, self.calibrated_face_center, 
                    self.calibrated_face_size, self.face_roi_scale
                ):
                    self.driver_absent_frame_count += 1
                    if self.driver_absent_frame_count >= self.driver_absent_frame_threshold:
                        results["is_driver_present"] = False
                    else:
                        results["is_driver_present"] = True
                    results["is_distracted_no_face"] = True
                    # ROI 범위를 벗어난 얼굴은 랜드마크를 None으로 설정
                    results["face_landmarks"] = None
                    return results
            
            # ROI 필터링이 비활성화되었거나 ROI 범위 내의 얼굴 처리
            self.driver_absent_frame_count = 0
            results["is_driver_present"] = True
            results["face_landmarks"] = face_result.face_landmarks[0]
            
            # --- Pupil-based Gaze Detection ---
            pupil_gaze_deviated = calculate_mediapipe_pupil_gaze_deviation(
                face_result.face_landmarks[0], frame_size, self.calibrated_pupil_center,
                self.pupil_gaze_threshold, self.pupil_gaze_consec_frames
            )
            results["is_pupil_gaze_deviated"] = pupil_gaze_deviated
            
            # 진짜 pitch 계산
            true_pitch = get_mediapipe_true_pitch_from_landmarks(face_result.face_landmarks[0])
            results["true_pitch"] = true_pitch
            # true_pitch 캘리브레이션 적용
            if not hasattr(self, 'calibrated_true_pitch'):
                self.calibrated_true_pitch = true_pitch
            if self.just_calibrated:
                self.calibrated_true_pitch = true_pitch
                # print(f"[MediaPipeAnalyzer] True pitch calibrated: {true_pitch:.2f}")
            true_pitch_diff = true_pitch - self.calibrated_true_pitch
            # print(f"[DEBUG] True pitch (nose-eye): {true_pitch:.2f} deg, diff={true_pitch_diff:.2f}")
            # true_pitch로 고개 숙임 감지
            if abs(true_pitch_diff) > TRUE_PITCH_THRESHOLD:
                results["is_head_down"] = True
                print(f"[DEBUG] True pitch head down detected: |diff|={abs(true_pitch_diff):.2f} > {TRUE_PITCH_THRESHOLD}")
            else:
                results["is_head_down"] = False
            
            # true_pitch 값도 결과에 추가 (Visualizer에서 표시용)
            results["true_pitch"] = true_pitch
            results["true_pitch_diff"] = true_pitch_diff
            results["true_pitch_threshold"] = TRUE_PITCH_THRESHOLD
            if face_result.face_blendshapes:
                blendshapes = {
                    b.category_name: b.score for b in face_result.face_blendshapes[0]
                }
                results["all_blendshapes"] = blendshapes
                
                # 먼저 head pose를 계산하여 동적 임계값 결정
                current_pitch = 0.0
                if face_result.facial_transformation_matrixes:
                    head_pose = get_mediapipe_head_pose_from_matrix(
                        face_result.facial_transformation_matrixes[0]
                    )
                    pitch, yaw, roll = head_pose
                    
                    # 방금 캘리브레이션 완료된 경우 현재 head pose 값을 캘리브레이션 오프셋으로 설정
                    if self.just_calibrated:
                        self.mp_front_face_offset_pitch = -pitch
                        self.mp_front_face_offset_yaw = -yaw
                        self.mp_front_face_offset_roll = -roll
                        self.just_calibrated = False  # 플래그 초기화
                        print(f"[MediaPipeAnalyzer] Head pose offsets updated after calibration: pitch={-pitch:.1f}, yaw={-yaw:.1f}, roll={-roll:.1f}")
                    
                    # 캘리브레이션된 오프셋 적용
                    current_pitch = pitch + self.mp_front_face_offset_pitch
                    current_yaw = yaw + self.mp_front_face_offset_yaw
                    current_roll = roll + self.mp_front_face_offset_roll
                    
                    results["mp_head_pitch_deg"] = current_pitch
                    results["mp_head_yaw_deg"] = current_yaw
                    results["mp_head_roll_deg"] = current_roll
                
                # 고개 각도에 따른 동적 눈 감음 임계값 결정
                if current_pitch < HEAD_UP_THRESHOLD_FOR_EYE:
                    # 고개를 들었을 때 - 더 관대한 임계값
                    dynamic_eye_threshold = EYE_BLINK_THRESHOLD_HEAD_UP
                    # print(f"[DEBUG] Head up detected (pitch={current_pitch:.1f}°), using lower eye threshold: {dynamic_eye_threshold}")
                elif current_pitch > HEAD_DOWN_THRESHOLD_FOR_EYE:
                    # 고개를 숙였을 때 - 중간 임계값
                    dynamic_eye_threshold = EYE_BLINK_THRESHOLD_HEAD_DOWN
                    # print(f"[DEBUG] Head down detected (pitch={current_pitch:.1f}°), using medium eye threshold: {dynamic_eye_threshold}")
                else:
                    # 정면일 때 - 기본 임계값
                    dynamic_eye_threshold = EYE_BLINK_THRESHOLD
                    # print(f"[DEBUG] Head normal (pitch={current_pitch:.1f}°), using default eye threshold: {dynamic_eye_threshold}")
                
                # 동적 임계값을 사용한 눈 감음 감지
                left_eye_blink, right_eye_blink = (
                    blendshapes.get("eyeBlinkLeft", 0),
                    blendshapes.get("eyeBlinkRight", 0),
                )
                avg_blink = (left_eye_blink + right_eye_blink) / 2.0
                results["mp_ear"] = avg_blink  # blendshapes 값 그대로 사용
                results["ear_value"] = avg_blink  # 기존 호환성 유지
                results["dynamic_eye_threshold"] = dynamic_eye_threshold  # 디버깅용
                
                self.eye_closed_frame_count = (
                    self.eye_closed_frame_count + 1
                    if avg_blink > dynamic_eye_threshold
                    else 0
                )
                if self.eye_closed_frame_count >= DROWSY_CONSEC_FRAMES:
                    results["is_drowsy"] = True
                jaw_open = blendshapes.get("jawOpen", 0)
                results["mar_value"] = jaw_open
                self.yawn_frame_count = (
                    self.yawn_frame_count + 1 if jaw_open > JAW_OPEN_THRESHOLD else 0
                )
                if self.yawn_frame_count >= YAWN_CONSEC_FRAMES:
                    results["is_yawning"] = True
                gaze_right = blendshapes.get("eyeLookOutRight", 0) + blendshapes.get(
                    "eyeLookInLeft", 0
                )
                gaze_left = blendshapes.get("eyeLookOutLeft", 0) + blendshapes.get(
                    "eyeLookInRight", 0
                )
                gaze_up, gaze_down = (
                    blendshapes.get("eyeLookUpLeft", 0)
                    + blendshapes.get("eyeLookUpRight", 0),
                    blendshapes.get("eyeLookDownLeft", 0)
                    + blendshapes.get("eyeLookDownRight", 0),
                )
                gaze_x, gaze_y = (
                    (gaze_left - gaze_right) / 2.0,
                    (gaze_down - gaze_up) / 2.0,
                )
                results["gaze_vector"] = (gaze_x, gaze_y)
                results["gaze_x"] = gaze_x
                results["gaze_y"] = gaze_y
                
                # 방금 캘리브레이션 완료된 경우 현재 gaze 값을 캘리브레이션 값으로 업데이트
                if self.just_calibrated:
                    self.calibrated_gaze_x = gaze_x
                    self.calibrated_gaze_y = gaze_y
                    print(f"[MediaPipeAnalyzer] Gaze updated after calibration: x={gaze_x:.3f}, y={gaze_y:.3f}")
                # 자동 캘리브레이션 제거 - 사용자가 Calibrate 버튼을 클릭할 때만 캘리브레이션
                # elif not self.is_calibrated:
                #     self.calibrated_gaze_x = gaze_x
                #     self.calibrated_gaze_y = gaze_y
                #     self.is_calibrated = True  # 캘리브레이션 완료 표시
                #     print(f"[MediaPipeAnalyzer] Gaze auto-calibrated: x={gaze_x:.3f}, y={gaze_y:.3f}")
                
                # 시선 이탈 감지 (캘리브레이션 적용)
                gaze_diff = abs(gaze_x - self.calibrated_gaze_x)
                if gaze_diff > GAZE_THRESHOLD:
                    results["is_gaze"] = True
                else:
                    pass
                
                # Head pose compensation for gaze detection
                # 고개 회전을 보정하여 실제 시선 방향 계산
                if face_result.facial_transformation_matrixes:
                    head_pose = get_mediapipe_head_pose_from_matrix(
                        face_result.facial_transformation_matrixes[0]
                    )
                    pitch, yaw, roll = head_pose
                    
                    # 고개 회전 각도를 라디안으로 변환
                    yaw_rad = np.radians(yaw)
                    pitch_rad = np.radians(pitch)
                    
                    # 고개 회전을 보정한 gaze 계산
                    # Yaw 회전 보정 (좌우 회전)
                    compensated_gaze_x = gaze_x * np.cos(yaw_rad) - gaze_y * np.sin(yaw_rad)
                    compensated_gaze_y = gaze_x * np.sin(yaw_rad) + gaze_y * np.cos(yaw_rad)
                    
                    # Pitch 회전 보정 (상하 회전) - 간단한 근사
                    compensated_gaze_y = compensated_gaze_y * np.cos(pitch_rad)
                    
                    # 보정된 gaze 값 저장
                    results["compensated_gaze_x"] = compensated_gaze_x
                    results["compensated_gaze_y"] = compensated_gaze_y
                    
                    # 간단한 해결책: 고개가 많이 돌아가면 gaze 감지 비활성화
                    if abs(yaw) < HEAD_ROTATION_THRESHOLD_FOR_GAZE:
                        # 고개가 거의 정면일 때만 gaze 감지
                        gaze_diff = abs(gaze_x - self.calibrated_gaze_x)
                        if gaze_diff > GAZE_THRESHOLD:
                            results["is_gaze_compensated"] = True
                            results["is_gaze"] = True
                        else:
                            results["is_gaze_compensated"] = False
                            results["is_gaze"] = False
                    else:
                        # 고개가 많이 돌아가면 gaze 감지 비활성화
                        results["is_gaze_compensated"] = False
                        results["is_gaze"] = False
                        results["gaze_disabled_due_to_head_rotation"] = True
                    
                    # print(f"[DEBUG] Gaze compensation: raw({gaze_x:.3f}, {gaze_y:.3f}) -> compensated({compensated_gaze_x:.3f}, {compensated_gaze_y:.3f})")
                    # print(f"[DEBUG] Head pose for compensation: yaw={yaw:.1f}°, pitch={pitch:.1f}°")
                    # print(f"[DEBUG] Gaze detection: {'enabled' if abs(yaw) < HEAD_ROTATION_THRESHOLD_FOR_GAZE else 'disabled (head rotated)'}")
                
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
                self.gaze_deviated_frame_count = (
                    self.gaze_deviated_frame_count + 1
                    if gaze_magnitude > GAZE_VECTOR_THRESHOLD
                    else 0
                )
                if self.gaze_deviated_frame_count >= POSE_CONSEC_FRAMES:
                    results["is_gaze_deviated"] = True
                    
            if face_result.facial_transformation_matrixes:
                head_pose = get_mediapipe_head_pose_from_matrix(
                    face_result.facial_transformation_matrixes[0]
                )
                results["head_pose"] = head_pose
                pitch, yaw, roll = head_pose
                
                # 방금 캘리브레이션 완료된 경우 현재 head pose 값을 캘리브레이션 오프셋으로 설정
                if self.just_calibrated:
                    self.mp_front_face_offset_pitch = -pitch
                    self.mp_front_face_offset_yaw = -yaw
                    self.mp_front_face_offset_roll = -roll
                    self.just_calibrated = False  # 플래그 초기화 (gaze와 head pose 모두 처리 후)
                    print(f"[MediaPipeAnalyzer] Head pose offsets updated after calibration: pitch={-pitch:.1f}, yaw={-yaw:.1f}, roll={-roll:.1f}")
                
                # 캘리브레이션된 오프셋 적용
                current_pitch = pitch + self.mp_front_face_offset_pitch
                current_yaw = yaw + self.mp_front_face_offset_yaw
                current_roll = roll + self.mp_front_face_offset_roll
                
                # 디버깅 출력 (캘리브레이션 상태 확인)
                # print(f"[DEBUG] Head pose: raw(pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f})")
                # print(f"[DEBUG] Head pose: offsets(pitch={self.mp_front_face_offset_pitch:.1f}, yaw={self.mp_front_face_offset_yaw:.1f}, roll={self.mp_front_face_offset_roll:.1f})")
                # print(f"[DEBUG] Head pose: calibrated(pitch={current_pitch:.1f}, yaw={current_yaw:.1f}, roll={current_roll:.1f})")
                
                results["mp_head_pitch_deg"] = current_pitch
                results["mp_head_yaw_deg"] = current_yaw
                results["mp_head_roll_deg"] = current_roll
                
                # Head pose color based on calibration status (similar to Dlib)
                if not self.is_calibrated:
                    results["mp_head_pose_color"] = (100, 100, 100)  # Grey if not calibrated
                    # 캘리브레이션 전에는 값을 0으로 설정
                    results["mp_head_pitch_deg"] = 0.0
                    results["mp_head_yaw_deg"] = 0.0
                    results["mp_head_roll_deg"] = 0.0
                    results["is_head_down"] = False
                    results["is_head_up"] = False
                    results["is_distracted_from_front"] = False
                    results["mp_is_distracted_from_front"] = False
                else:
                    # Distracted detection이 비활성화된 경우 검출하지 않음
                    if not self.enable_distracted_detection:
                        results["mp_head_pose_color"] = (100, 100, 100)  # Grey when disabled
                        results["is_distracted_from_front"] = False
                        results["mp_is_distracted_from_front"] = False
                        results["is_look_ahead_warning"] = False
                    else:
                        # 정면 이탈 감지 (운전 상황에 맞게 조정)
                        if (abs(current_yaw) > MP_YAW_THRESHOLD or
                            abs(current_pitch) > MP_PITCH_THRESHOLD):
                            results["is_distracted_from_front"] = True
                            results["mp_is_distracted_from_front"] = True
                            results["mp_head_pose_color"] = (0, 0, 255)  # Red for distracted
                            # print(f"[DEBUG] Head pose deviated: yaw={abs(current_yaw):.1f}>({MP_YAW_THRESHOLD}), pitch={abs(current_pitch):.1f}>({MP_PITCH_THRESHOLD})")
                        else:
                            results["mp_head_pose_color"] = (0, 255, 0)  # Green for normal
                            # print(f"[DEBUG] Head pose normal: yaw={abs(current_yaw):.1f}<={MP_YAW_THRESHOLD}, pitch={abs(current_pitch):.1f}<={MP_PITCH_THRESHOLD}")
                    
                    # 고개 숙임/들기 감지 (true_pitch 방식이 이미 처리했으므로 여기서는 제거)
                    # true_pitch 방식이 더 정확하므로 일반 head pose 방식은 사용하지 않음
                    # results["is_head_down"]은 이미 true_pitch 방식에서 설정됨
                    # results["is_head_up"]만 여기서 설정
                    if current_pitch < PITCH_UP_THRESHOLD:
                        results["is_head_up"] = True
                    else:
                        results["is_head_up"] = False
                
                # 위험 상태 감지: 눈을 감은 상태에서 고개가 숙여지는 경우
                if results["is_drowsy"] and results["is_head_down"]:
                    results["is_dangerous_condition"] = True
                    results["dangerous_condition_message"] = "DANGER: Eyes Closed + Head Down!"
                    # 캘리브레이션된 경우에만 메시지 출력
                    if self.is_calibrated:
                        print("[MediaPipeAnalyzer] DANGEROUS CONDITION DETECTED: Eyes closed and head down!")
        
        # --- 눈 감김 비율 계산 및 추가 ---
        # (예시: 평균 눈높이/눈뜨기 최대값 등으로 계산)
        if face_result and face_result.face_landmarks:
            lm = face_result.face_landmarks[0].landmark if hasattr(face_result.face_landmarks[0], 'landmark') else face_result.face_landmarks[0]
            left_eye_height = abs(lm[159].y - lm[145].y)
            right_eye_height = abs(lm[386].y - lm[374].y)
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            max_open = 0.3
            eye_closed_ratio = 1.0 - min(avg_eye_height / max_open, 1.0)
            results["eye_closed_ratio"] = eye_closed_ratio
            # config의 eye_blink_threshold도 함께 전달
            results["eye_blink_threshold"] = get_mediapipe_config("eye_blink_threshold", 0.25)

        # --- Hand Analysis ---
        if not self.enable_hand_detection or self.hand_landmarker is None:
            # 손 감지 완전 비활성화
            results["hand_status"] = "Hand Detection Disabled"
            results["hand_warning_color"] = "green"
            results["hand_warning_message"] = "Hand detection is OFF"
            results["is_left_hand_off"] = False
            results["is_right_hand_off"] = False
            results["are_both_hands_on_wheel"] = True
        else:
            results["is_left_hand_off"] = False
            results["is_right_hand_off"] = False
            
            left_hand_detected = False
            right_hand_detected = False

            # 얼굴 랜드마크 가져오기 (크기 비교용)
            face_landmarks = None
            if face_result and face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]

            if hand_result and hand_result.hand_landmarks:
                for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    if len(hand_landmarks) > 0:
                        # 손 크기 필터링 확인
                        if not self._should_ignore_hand(hand_landmarks, face_landmarks):
                            if i == 0:  # 첫 번째 손
                                left_hand_detected = True
                                results["left_hand_landmarks"] = hand_landmarks
                            elif i == 1:  # 두 번째 손
                                right_hand_detected = True
                                results["right_hand_landmarks"] = hand_landmarks
            
            # 손 감지 상태에 따른 경고 설정 (로직 수정)
            if left_hand_detected or right_hand_detected:
                # 손이 감지됨
                if left_hand_detected and right_hand_detected:
                    # 두 손이 모두 감지됨 - 연속 프레임 카운터 증가
                    self.hand_detected_frame_count += 1
                    
                    # 60프레임 이상 연속 두 손 감지 시 경고
                    if self.hand_detected_frame_count >= self.hand_warning_frame_threshold:
                        results["hand_status"] = "Both Hands Detected - WARNING!"
                        results["hand_warning_color"] = "red"
                        results["hand_warning_message"] = "Please hold the steering wheel!"
                        results["is_left_hand_off"] = True
                        results["is_right_hand_off"] = True
                    else:
                        # 60프레임 미만이면 정상 상태로 표시
                        results["hand_status"] = "Both Hands Detected - GOOD"
                        results["hand_warning_color"] = "green"
                        results["hand_warning_message"] = "Hands on wheel - GOOD"
                        results["is_left_hand_off"] = False
                        results["is_right_hand_off"] = False
                else:
                    # 한 손만 감지됨 - 정상 상태로 표시하고 카운터 리셋
                    self.hand_detected_frame_count = 0
                    results["hand_status"] = "One Hand Detected - GOOD"
                    results["hand_warning_color"] = "green"
                    results["hand_warning_message"] = "Hands on wheel - GOOD"
                    if left_hand_detected:
                        results["is_left_hand_off"] = False
                    if right_hand_detected:
                        results["is_right_hand_off"] = False
                
                # 캘리브레이션된 경우에만 메시지 출력
                if self.is_calibrated:
                    if left_hand_detected and right_hand_detected:
                        if self.hand_detected_frame_count >= self.hand_warning_frame_threshold:
                            print("[MediaPipeAnalyzer] Both hands detected for 60+ frames - RED WARNING")
                        else:
                            print("[MediaPipeAnalyzer] Both hands detected - GREEN (GOOD)")
                    else:
                        print("[MediaPipeAnalyzer] One hand detected - GREEN (GOOD)")
            else:
                # 손이 감지되지 않음 - 연속 프레임 카운터 리셋
                self.hand_detected_frame_count = 0
                
                # 손이 감지되지 않음 - 초록색 (정상)
                results["hand_status"] = "No Hands Detected - GOOD"
                results["hand_warning_color"] = "green"
                results["hand_warning_message"] = "Hands on wheel - GOOD"
                results["is_left_hand_off"] = False
                results["is_right_hand_off"] = False
                # 캘리브레이션된 경우에만 메시지 출력
                if self.is_calibrated:
                    print("[MediaPipeAnalyzer] No hands detected - GREEN (GOOD)")
            
            # 손 감지 연속 프레임 수를 결과에 추가
            results["hand_detected_frame_count"] = self.hand_detected_frame_count
            results["hand_warning_frame_threshold"] = self.hand_warning_frame_threshold

        if results["is_drowsy"]:
            self.drowsy_frame_count += 1
        else:
            self.drowsy_frame_count = 0
        results["drowsy_frame_count"] = self.drowsy_frame_count

        if results["is_distracted_from_front"]:
            self.distracted_frame_count += 1
        else:
            self.distracted_frame_count = 0
        results["distracted_frame_count"] = self.distracted_frame_count

        # ROI 정보 추가
        if self.enable_face_position_filtering:
            results["face_roi_bounds"] = self.calibrated_face_roi

        return results

    def detect_sync(self, frame):  # IMAGE, VIDEO 모드용 동기 함수
        if not self.use_task_api:
            return self._analyze_frame_traditional(frame)
        
        # 실제 프레임 크기를 저장하여 _process_results에서 사용
        self.current_frame_size = (frame.shape[0], frame.shape[1])
            
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        face_result = (
            self.face_landmarker.detect(mp_image)
            if self.running_mode == vision.RunningMode.IMAGE
            else self.face_landmarker.detect_for_video(
                mp_image, int(time.time() * 1000)
            )
        )
        hand_result = None
        if self.hand_landmarker is not None:
            hand_result = (
                self.hand_landmarker.detect(mp_image)
                if self.running_mode == vision.RunningMode.IMAGE
                else self.hand_landmarker.detect_for_video(
                    mp_image, int(time.time() * 1000)
                )
            )
        return self._process_results(face_result, hand_result)

    def detect_async(self, frame, timestamp_ms):  # LIVE_STREAM 모드용 비동기 함수
        if not self.use_task_api:
            return
            
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        self.face_landmarker.detect_async(mp_image, timestamp_ms)
        if self.hand_landmarker is not None:
            self.hand_landmarker.detect_async(mp_image, timestamp_ms)

    def _calculate_face_size(self, face_landmarks):
        """얼굴 크기를 계산합니다 (면적 기준)"""
        if not face_landmarks:
            return 0.0
        
        # 얼굴의 주요 포인트들을 사용하여 얼굴 크기 계산
        # 코, 눈, 입의 좌표를 사용
        try:
            if hasattr(face_landmarks, 'landmark'):
                # 기존 Face Mesh 방식
                landmarks = face_landmarks.landmark
                # 얼굴 윤곽선 포인트들 (0-16)
                face_contour = [(landmarks[i].x, landmarks[i].y) for i in range(17)]
            else:
                # Task API 방식
                landmarks = face_landmarks
                face_contour = [(landmarks[i].x, landmarks[i].y) for i in range(17)]
            
            # 얼굴 윤곽선의 바운딩 박스 계산
            x_coords = [p[0] for p in face_contour]
            y_coords = [p[1] for p in face_contour]
            
            face_width = max(x_coords) - min(x_coords)
            face_height = max(y_coords) - min(y_coords)
            face_area = face_width * face_height
            
            return face_area
        except Exception as e:
            print(f"[MediaPipeAnalyzer] Error calculating face size: {e}")
            return 0.0

    def _calculate_hand_size(self, hand_landmarks):
        """손 크기를 계산합니다 (면적 기준)"""
        if not hand_landmarks:
            return 0.0
        
        try:
            if hasattr(hand_landmarks, 'landmark'):
                # 기존 방식
                landmarks = hand_landmarks.landmark
                hand_points = [(landmarks[i].x, landmarks[i].y) for i in range(21)]
            else:
                # Task API 방식
                landmarks = hand_landmarks
                hand_points = [(landmarks[i].x, landmarks[i].y) for i in range(21)]
            
            # 손의 바운딩 박스 계산
            x_coords = [p[0] for p in hand_points]
            y_coords = [p[1] for p in hand_points]
            
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hand_area = hand_width * hand_height
            
            return hand_area
        except Exception as e:
            print(f"[MediaPipeAnalyzer] Error calculating hand size: {e}")
            return 0.0

    def _should_ignore_hand(self, hand_landmarks, face_landmarks):
        """손 크기가 얼굴 크기의 2/3보다 작으면 무시할지 결정하고, 화면 하단 1/3 영역에 있는지 확인"""
        if not ENABLE_HAND_SIZE_FILTERING:
            return False
        
        if not hand_landmarks:
            return False
        
        # 손이 화면 하단 1/3 영역에 있는지 확인
        if hasattr(hand_landmarks, 'landmark'):
            landmarks = hand_landmarks.landmark
            hand_points = [(landmarks[i].x, landmarks[i].y) for i in range(21)]
        else:
            landmarks = hand_landmarks
            hand_points = [(landmarks[i].x, landmarks[i].y) for i in range(21)]
        
        # 손의 중심점 계산
        hand_center_x = np.mean([p[0] for p in hand_points])
        hand_center_y = np.mean([p[1] for p in hand_points])
        
        # 화면 하단 1/3 영역 (y > 0.67)에 있는지 확인
        if hand_center_y < 0.67:  # 정규화된 좌표에서 0.67 = 화면의 2/3 지점
            if self.is_calibrated:
                print(f"[MediaPipeAnalyzer] Hand ignored: outside bottom 1/3 area (y={hand_center_y:.3f} < 0.67)")
            return True
        
        # 얼굴 랜드마크가 있는 경우 크기 비교도 수행
        if face_landmarks:
            hand_area = self._calculate_hand_size(hand_landmarks)
            face_area = self._calculate_face_size(face_landmarks)
            
            if face_area == 0.0:
                return False
            
            hand_face_ratio = hand_area / face_area
            should_ignore = hand_face_ratio < HAND_SIZE_RATIO_THRESHOLD
            
            if should_ignore and self.is_calibrated:
                print(f"[MediaPipeAnalyzer] Hand ignored: ratio={hand_face_ratio:.3f} < {HAND_SIZE_RATIO_THRESHOLD}")
            
            return should_ignore
        
        return False

    def _is_face_within_calibrated_bounds(self, face_landmarks, frame_size):
        """
        Check if the current detected face is within the calibrated driver's position and size bounds.
        Returns True if the face is likely the same driver, False if it's a different person.
        """
        if not self.enable_face_position_filtering:
            return True  # If filtering is disabled, accept all faces
        
        if not self.is_calibrated or self.calibrated_face_center is None or self.calibrated_face_size is None:
            return True  # If not calibrated, accept all faces
        
        # Calculate current face center and size using nose as reference point
        if hasattr(face_landmarks, 'landmark'):
            # 기존 Face Mesh 방식
            landmarks = face_landmarks.landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            # 코 위치를 기준점으로 사용 (랜드마크 인덱스 1이 코)
            nose_x = landmarks[1].x * frame_size[1]
            nose_y = landmarks[1].y * frame_size[0]
        else:
            # Task API 방식 - 리스트 형태
            x_coords = [lm.x for lm in face_landmarks]
            y_coords = [lm.y for lm in face_landmarks]
            # 코 위치를 기준점으로 사용 (랜드마크 인덱스 1이 코)
            nose_x = face_landmarks[1].x * frame_size[1]
            nose_y = face_landmarks[1].y * frame_size[0]
        
        # 코를 중심으로 하는 얼굴 크기 계산
        current_face_center_x = nose_x
        current_face_center_y = nose_y
        current_face_width = (max(x_coords) - min(x_coords)) * frame_size[1]
        current_face_height = (max(y_coords) - min(y_coords)) * frame_size[0]
        
        # Check position deviation
        calibrated_center_x, calibrated_center_y = self.calibrated_face_center
        calibrated_width, calibrated_height = self.calibrated_face_size
        
        # 새로운 방식: 캘리브레이션된 얼굴 ROI의 config에서 설정된 배수로 detection 영역 설정
        roi_width = calibrated_width * self.face_roi_scale
        roi_height = calibrated_height * self.face_roi_scale
        
        # ROI의 경계 계산
        roi_x1 = calibrated_center_x - roi_width / 2
        roi_y1 = calibrated_center_y - roi_height / 2
        roi_x2 = calibrated_center_x + roi_width / 2
        roi_y2 = calibrated_center_y + roi_height / 2
        
        # 현재 얼굴이 ROI 내에 있는지 확인
        current_face_x1 = current_face_center_x - current_face_width / 2
        current_face_y1 = current_face_center_y - current_face_height / 2
        current_face_x2 = current_face_center_x + current_face_width / 2
        current_face_y2 = current_face_center_y + current_face_height / 2
        
        # 얼굴이 ROI와 겹치는지 확인
        face_in_roi = (current_face_x1 < roi_x2 and current_face_x2 > roi_x1 and
                      current_face_y1 < roi_y2 and current_face_y2 > roi_y1)
        
        if not face_in_roi:
            print(f"[MediaPipeAnalyzer] Face rejected - outside ROI bounds")
            print(f"[MediaPipeAnalyzer] ROI: ({roi_x1:.1f}, {roi_y1:.1f}) to ({roi_x2:.1f}, {roi_y2:.1f})")
            print(f"[MediaPipeAnalyzer] Face: ({current_face_x1:.1f}, {current_face_y1:.1f}) to ({current_face_x2:.1f}, {current_face_y2:.1f})")
            return False
        
        return True


