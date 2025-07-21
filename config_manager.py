# config_manager.py

import json
import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일을 로드합니다."""
        if not self.config_file.exists():
            print(f"Warning: Config file {self.config_file} not found. Using default values.")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Config loaded from {self.config_file}")
            return config
        except Exception as e:
            print(f"Error loading config file: {e}. Using default values.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정값을 반환합니다."""
        return {
            "dlib": {
                "eye_ar_thresh": 0.15,
                "eye_ar_consec_frames": 15,
                "mouth_ar_thresh": 0.4,
                "mouth_ar_consec_frames": 30,
                "pitch_threshold": 15.0,
                "yaw_threshold": 60.0,
                "roll_threshold": 90.0,
                "pitch_down_threshold": 10.0,
                "distraction_consec_frames": 10
            },
            "mediapipe": {
                "eye_blink_threshold": 0.3,
                "eye_blink_threshold_head_up": 0.2,
                "eye_blink_threshold_head_down": 0.25,
                "head_up_threshold_for_eye": -10.0,
                "head_down_threshold_for_eye": 8.0,
                "jaw_open_threshold": 0.4,
                "drowsy_consec_frames": 15,
                "yawn_consec_frames": 30,
                "pitch_down_threshold": 10,
                "pitch_up_threshold": -15,
                "pose_consec_frames": 20,
                "gaze_vector_threshold": 0.5,
                "mp_yaw_threshold": 30.0,
                "mp_pitch_threshold": 10.0,
                "mp_roll_threshold": 999.0,
                "gaze_threshold": 0.5,
                "distraction_consec_frames": 10,
                "true_pitch_threshold": 10.0,
                "head_rotation_threshold_for_gaze": 15.0,
                "use_video_mode": True,
                "min_hand_detection_confidence": 0.3,
                "min_hand_presence_confidence": 0.3,
                "hand_off_consec_frames": 5,
                "hand_size_ratio_threshold": 0.67,
                "enable_hand_size_filtering": True
            },
            "3ddfa": {
                "eye_ar_thresh": 0.22,
                "mouth_ar_thresh": 0.6
            },
            "yolo": {
                "default_conf_thres": 0.25,
                "default_iou_thres": 0.45,
                "default_max_det": 1000
            },
            "general": {
                "fps_display": True,
                "debug_mode": False
            },
            "openvino": {
                "ear_threshold": 0.2,
                "mar_threshold": 0.4,
                "eye_ar_consec_frames": 20,
                "mouth_ar_consec_frames": 40,
                "head_pose_threshold": 12.0,
                "head_down_consec_frames": 8,
                "distraction_consec_frames": 15,
                "distraction_yaw_threshold": 30.0,
                "distraction_roll_threshold": 45.0,
                "frame_skip": 3,
                "face_detection_cache_time": 0.15,
                "target_fps": 20.0,
                "max_frame_skip": 2,
                "ear_history_length": 30,
                "mar_history_length": 30,
                "head_pose_history_length": 30,
                "ear_variance_threshold": 0.001,
                "mar_variance_threshold": 0.01,
                "calibration_ear_ratio": 0.8,
                "use_hybrid_mode": True,
                "device": "CPU",
                "conf_thres": 0.5
            }
        }
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """설정값을 가져옵니다."""
        try:
            return self.config[section][key]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"Config key '{section}.{key}' not found")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """섹션 전체를 가져옵니다."""
        try:
            return self.config[section].copy()
        except KeyError:
            raise KeyError(f"Config section '{section}' not found")
    
    def set(self, section: str, key: str, value: Any) -> None:
        """설정값을 설정합니다."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save(self) -> None:
        """설정을 파일에 저장합니다."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"Config saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def reload(self) -> None:
        """설정 파일을 다시 로드합니다."""
        self.config = self._load_config()
    
    def print_config(self) -> None:
        """현재 설정을 출력합니다."""
        print("Current Configuration:")
        print(json.dumps(self.config, indent=2, ensure_ascii=False))

# 전역 설정 매니저 인스턴스
config_manager = ConfigManager()

# 편의 함수들
def get_dlib_config(key: str, default: Any = None) -> Any:
    """Get Dlib configuration value"""
    return config_manager.get("dlib", key, default)

def get_mediapipe_config(key: str, default: Any = None) -> Any:
    """Get MediaPipe configuration value"""
    return config_manager.get("mediapipe", key, default)

def get_3ddfa_config(key: str, default: Any = None) -> Any:
    """Get 3DDFA configuration value"""
    return config_manager.get("3ddfa", key, default)

def get_yolo_config(key: str, default: Any = None) -> Any:
    """Get YOLO configuration value"""
    return config_manager.get("yolo", key, default)

def get_general_config(key: str, default: Any = None) -> Any:
    """Get general configuration value"""
    return config_manager.get("general", key, default) 