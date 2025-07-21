# visualizer.py

import cv2
import numpy as np
from imutils import face_utils
import mediapipe as mp
import mediapipe.framework.formats.landmark_pb2 

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

# 랜드마크(점)와 연결선 스타일 지정 (원하는 색상으로 설정 가능)
face_landmark_spec = mp_drawing.DrawingSpec(color=(144, 238, 144), thickness=1, circle_radius=1) # 밝은 초록
face_connection_spec = mp_drawing.DrawingSpec(color=(144, 238, 144), thickness=1, circle_radius=1)

# 핸드랜드마크 스타일
hand_landmark_spec = mp_drawing.DrawingSpec(color=(255, 128, 0), thickness=2, circle_radius=2) # 주황색
hand_connection_spec = mp_drawing.DrawingSpec(color=(255, 200, 100), thickness=2) # 밝은 주황색

# 포즈 랜드마크 스타일
pose_landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2) # 시안색
pose_connection_spec = mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2) # 어두운 시안색

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
        self.line_thickness = 3
        self.text_x_align = 10
        self.fps_y = 30
        self.yolo_info_start_y = 60
        self.dlib_info_start_y = 50
        self.mediapipe_info_start_y = 200
        self.text_spacing = 30
        # Dlib, MediaPipe 정면 상태 표시 위치 조정
        self.dlib_front_status_y = self.dlib_info_start_y + 5 * self.text_spacing
        self.mediapipe_front_status_y = self.mediapipe_info_start_y + 4 * self.text_spacing
        # Crop offset 설정
        self.crop_offset = 0

    def draw_yolov5_results(self, frame, detections, names, hide_labels=False, hide_conf=False):
        tl = self.line_thickness
        h, w = frame.shape[:2]
        
        # YOLO 상태를 좌측 제일 위에 표시
        yolo_status = "YOLO: No Detection"
        yolo_color = (100, 100, 100)  # 회색
        
        if detections is not None and len(detections):
            # 가장 높은 신뢰도의 detection을 사용
            best_detection = max(detections, key=lambda x: x[4])
            class_name = names[int(best_detection[5])]
            confidence = best_detection[4]
            
            if class_name == 'normal':
                yolo_status = f"YOLO: Normal ({confidence:.2f})"
                yolo_color = (255, 200, 90)
            elif class_name in ['drowsy', 'drowsy#2']:
                yolo_status = f"YOLO: Drowsy ({confidence:.2f})"
                yolo_color = (0, 0, 255)
            elif class_name == 'yawning':
                yolo_status = f"YOLO: Yawning ({confidence:.2f})"
                yolo_color = (51, 255, 255)
        
        # 좌측 제일 위에 YOLO 상태 표시 (0.5 크기) - crop offset 적용
        y_pos = 30 + self.crop_offset  # 양수일 때 아래로 이동
        cv2.putText(frame, yolo_status, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolo_color, 2)
        
        if detections is not None and len(detections):
            for *xyxy, conf, cls in reversed(detections):
                yolo_color = (255, 200, 90)
                if names[int(cls)] == 'normal':
                    yolo_color = (255, 200, 90)
                elif names[int(cls)] in ['drowsy', 'drowsy#2']:
                    yolo_color = (0, 0, 255)
                elif names[int(cls)] == 'yawning':
                    yolo_color = (51, 255, 255)

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(frame, c1, c2, yolo_color, thickness=tl, lineType=cv2.LINE_AA)

                if not hide_labels:
                    label = (f"{names[int(cls)]} {conf:.2f}" if not hide_conf else names[int(cls)])
                    tf = max(tl - 1, 1)
                    t_size = cv2.getTextSize(label, 0, fontScale=self.font_scale, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(frame, c1, c2, yolo_color, -1, cv2.LINE_AA)
                    cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, self.font_scale, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return frame

    def draw_mediapipe_results(self,image, mp_display_results):
        
        # ----------------------------------------------------
        # 1. 얼굴 랜드마크 그리기 (mp_face_mesh.FACEMESH_TESSELATION 사용)
        # ----------------------------------------------------
        if mp_display_results.get("face_landmarks"):
            face_landmarks = mp_display_results["face_landmarks"]
            
            # Task API는 list 형태, 기존 Face Mesh는 NormalizedLandmarkList 객체
            if isinstance(face_landmarks, list):
                # Task API 형식: list of landmarks
                # 간단한 점으로만 표시 (연결선 없이)
                h, w, _ = image.shape
                for landmark in face_landmarks:
                    if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            else:
                # 기존 Face Mesh 형식: NormalizedLandmarkList 객체
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION, # 얼굴 메시 연결선 사용
                    landmark_drawing_spec=face_landmark_spec,
                    connection_drawing_spec=face_connection_spec
                )

        # ----------------------------------------------------
        # 2. 고개 자세 시각화 (nose_start_point2D와 nose_end_point2D 사용)
        # ----------------------------------------------------
        head_pose_full_data = mp_display_results.get('head_pose_data_full')

        if head_pose_full_data: # head_pose_full_data가 None이 아닌지 먼저 확인
            nose_start_point2D = head_pose_full_data.get('nose_start_point2D')
            nose_end_point2D = head_pose_full_data.get('nose_end_point2D')

            # 그리고 'nose_start_point2D'와 'nose_end_point2D'가 None이 아닌지, 그리고 (0,0)이 아닌지 확인
            if nose_start_point2D and nose_end_point2D and \
            nose_start_point2D != (0,0) and nose_end_point2D != (0,0):
                cv2.line(image, nose_start_point2D, nose_end_point2D, (255, 0, 0), 2) # 파란색 선으로 표시
        
        # ----------------------------------------------------
        # 3. 손 랜드마크 그리기
        # ----------------------------------------------------
        if mp_display_results.get("left_hand_landmarks"):
            left_hand_landmarks = mp_display_results["left_hand_landmarks"]
            if isinstance(left_hand_landmarks, list):
                # Task API 형식: 간단한 점으로 표시
                h, w, _ = image.shape
                for landmark in left_hand_landmarks:
                    if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            else:
                # 기존 형식
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=left_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec
                )
        
        if mp_display_results.get("right_hand_landmarks"):
            right_hand_landmarks = mp_display_results["right_hand_landmarks"]
            if isinstance(right_hand_landmarks, list):
                # Task API 형식: 간단한 점으로 표시
                h, w, _ = image.shape
                for landmark in right_hand_landmarks:
                    if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            else:
                # 기존 형식
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=right_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec
                )

        # ----------------------------------------------------
        # 4. 분석 결과 텍스트 오버레이 (오른쪽 제일 위에서부터 차례로 표시)
        # ----------------------------------------------------
        h, w, _ = image.shape
        # MediaPipe 텍스트를 오른쪽 제일 위에서부터 시작
        text_y_offset = 30  # 제일 위에서 30픽셀
        text_x_offset = w - 250  # 오른쪽에서 300픽셀 왼쪽
        text_spacing = 25  # 25픽셀 간격으로 통일
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0) # 초록색
        warning_color = (0, 0, 255) # 빨간색
        danger_color = (0, 0, 255) # 위험 상태용 빨간색
        
        # ⭐ 운전자 없음 메시지 최우선 표시
        if mp_display_results.get("is_driver_present") is False:
            cv2.putText(image, "No driver detected", (text_x_offset, text_y_offset), font, font_scale+0.3, warning_color, 3)
            text_y_offset += text_spacing
            return image
        
        # Danger/Warning 메시지 우선 표시 (제일 위)
        danger_y_offset = text_y_offset
        # 1. Danger: 눈감고 고개숙임
        if mp_display_results.get("is_dangerous_condition"):
            dangerous_message = mp_display_results.get("dangerous_condition_message", "DANGER: Eyes Closed + Head Down!")
            dangerous_message = dangerous_message.replace("MediaPipe: ", "").replace("Mediapipe: ", "")
            cv2.putText(image, dangerous_message, (text_x_offset, danger_y_offset), font, font_scale+0.3, (0,0,255), 3)
            danger_y_offset += text_spacing
        # 2. Wake UP
        drowsy_frame_count = mp_display_results.get("drowsy_frame_count", 0)
        wakeup_frame_threshold = mp_display_results.get("wakeup_frame_threshold", 60)
        if drowsy_frame_count >= wakeup_frame_threshold:
            cv2.putText(image, "Wake UP", (text_x_offset, danger_y_offset), font, font_scale+0.3, (0,0,255), 3)
            danger_y_offset += text_spacing
        
        # 3. Please look forward (Distracted detection이 활성화된 경우에만 표시)
        enable_distracted_detection = mp_display_results.get("enable_distracted_detection", True)
        if enable_distracted_detection:
            distracted_frame_count = mp_display_results.get("distracted_frame_count", 0)
            distracted_frame_threshold = mp_display_results.get("distracted_frame_threshold", 60)
            if distracted_frame_count >= distracted_frame_threshold:
                cv2.putText(image, "Please look forward", (text_x_offset, danger_y_offset), font, font_scale+0.3, (0,0,255), 3)
                danger_y_offset += text_spacing
        
        # 4. Please hold the steering wheel (두 손이 60프레임 이상 연속 감지된 경우)
        hand_detected_frame_count = mp_display_results.get("hand_detected_frame_count", 0)
        hand_warning_frame_threshold = mp_display_results.get("hand_warning_frame_threshold", 60)
        is_left_hand_off = mp_display_results.get("is_left_hand_off", False)
        is_right_hand_off = mp_display_results.get("is_right_hand_off", False)
        
        # 두 손이 모두 감지되어 60프레임 이상 연속일 때만 경고 표시
        if hand_detected_frame_count >= hand_warning_frame_threshold and is_left_hand_off and is_right_hand_off:
            cv2.putText(image, "Please hold the steering wheel", (text_x_offset, danger_y_offset), font, font_scale+0.3, (0,0,255), 3)
            danger_y_offset += text_spacing

        # 이후 일반 상태 메시지는 danger_y_offset 이후에 표시
        text_y_offset = danger_y_offset
        
        # ⭐ 현재 Head Pitch 값 상시 표시 (true_pitch 기반으로 변경)
        if 'true_pitch' in mp_display_results:
            # 실제 is_head_down 감지에 사용되는 true_pitch 값 사용
            true_pitch_val = mp_display_results['true_pitch']
            true_pitch_diff = mp_display_results.get('true_pitch_diff', 0.0)
            true_pitch_threshold = mp_display_results.get('true_pitch_threshold', 10.0)
            
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            
            # 캘리브레이션된 경우 diff 값을 표시, 아닌 경우 원본 값 표시
            is_calibrated = mp_display_results.get("mp_is_calibrated", False)
            if is_calibrated:
                # 캘리브레이션된 경우: diff 값이 0에 가까우면 정면
                display_value = true_pitch_diff
                cv2.putText(image, f"Pitch: {display_value:.1f} deg (calibrated, thresh: {true_pitch_threshold:.1f})", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            else:
                # 캘리브레이션되지 않은 경우: 원본 값 표시
                cv2.putText(image, f"Pitch: {true_pitch_val:.1f} deg (not calibrated, thresh: {true_pitch_threshold:.1f})", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += text_spacing
        elif 'mp_head_pitch_deg' in mp_display_results:
            # true_pitch가 없는 경우 기존 방식 사용
            pitch_val = mp_display_results['mp_head_pitch_deg']
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Pitch: {pitch_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += text_spacing
        
        # ⭐ (선택 사항) Yaw 값도 상시 표시하려면 추가
        if 'mp_head_yaw_deg' in mp_display_results:
            yaw_val = mp_display_results['mp_head_yaw_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Yaw: {yaw_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += text_spacing

        # ⭐ (선택 사항) Roll 값도 상시 표시하려면 추가
        if 'mp_head_roll_deg' in mp_display_results:
            roll_val = mp_display_results['mp_head_roll_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Roll: {roll_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += text_spacing
            
        # 캘리브레이션된 경우에만 상태 메시지 표시
        is_calibrated = mp_display_results.get("mp_is_calibrated", False)
        if is_calibrated:
        # 졸음 감지
            if mp_display_results.get("is_drowsy"):
                cv2.putText(image, "Drowsy!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += text_spacing

            # 하품 감지
            if mp_display_results.get("is_yawning"):
                cv2.putText(image, "Yawning!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += text_spacing

            # ⭐ 시선 이탈 감지 (gaze_x 기준)
            if mp_display_results.get("is_gaze"):
                cv2.putText(image, "Look Ahead", (text_x_offset, text_y_offset), font, font_scale, (0, 165, 255), 2)
                text_y_offset += text_spacing

            # ⭐ 눈동자 기반 시선 감지 결과 표시
            if mp_display_results.get("is_pupil_gaze_deviated"):
                cv2.putText(image, "Pupil Gaze: DEVIATED!", (text_x_offset, text_y_offset), 
                           font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            elif mp_display_results.get("enable_pupil_gaze_detection"):
                # 눈동자 기반 시선 감지가 활성화되어 있지만 이탈하지 않은 경우
                cv2.putText(image, "Pupil Gaze: OK", (text_x_offset, text_y_offset), 
                           font, font_scale, color, 2)
                text_y_offset += text_spacing

            # 보정된 시선 정보 표시 (새로 추가)
            if 'compensated_gaze_x' in mp_display_results and 'compensated_gaze_y' in mp_display_results:
                comp_gaze_x = mp_display_results['compensated_gaze_x']
                comp_gaze_y = mp_display_results['compensated_gaze_y']
                cv2.putText(image, f"Comp. Gaze: ({comp_gaze_x:.2f}, {comp_gaze_y:.2f})", 
                           (text_x_offset, text_y_offset), font, font_scale, (255, 255, 0), 2)
                text_y_offset += text_spacing
                
                # 보정된 시선 이탈 감지 표시
                if mp_display_results.get("is_gaze_compensated"):
                    cv2.putText(image, "Comp. Gaze: DEVIATED!", (text_x_offset, text_y_offset), 
                               font, font_scale, (0, 0, 255), 2)
                    text_y_offset += text_spacing
                    
            # Gaze 감지가 비활성화된 경우 표시
            if mp_display_results.get("gaze_disabled_due_to_head_rotation"):
                cv2.putText(image, "Gaze: DISABLED (Head Rotated)", (text_x_offset, text_y_offset), 
                           font, font_scale, (128, 128, 128), 2)  # 회색으로 표시
                text_y_offset += text_spacing

            # 전방 주시 이탈 (캘리브레이션된 경우)
            if mp_display_results.get("is_distracted_from_front"):
                cv2.putText(image, "Distracted from Front!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            
                # ----------------------------------------------------
                # 6. 새로운 손 감지 상태 표시 (MediaPipe 수정된 로직)
                # ----------------------------------------------------
                # 손 감지 상태에 따른 메시지와 색상 표시
                hand_status = mp_display_results.get("hand_status", "No Hands Detected")
                hand_warning_color = mp_display_results.get("hand_warning_color", "green")
                hand_warning_message = mp_display_results.get("hand_warning_message", "")
                
                # 색상 매핑
                color_map = {
                    "green": (0, 255, 0),    # 초록색
                    "yellow": (0, 255, 255), # 노란색 (BGR)
                    "red": (0, 0, 255)       # 빨간색
                }
            
                # 손 상태 표시
                if hand_warning_message:
                    display_color = color_map.get(hand_warning_color, (0, 255, 0))
                    cv2.putText(image, f"{hand_warning_message}", (text_x_offset, text_y_offset), 
                            font, font_scale, display_color, 2)
                    text_y_offset += text_spacing
                
                # 손 상태 정보 표시
                status_color = color_map.get(hand_warning_color, (0, 255, 0))
                cv2.putText(image, f"Hand Status: {hand_status}", (text_x_offset, text_y_offset), 
                        font, font_scale, status_color, 2)
                text_y_offset += text_spacing
            
                # 기존 손 이탈 감지 표시 (호환성을 위해 유지)
            if mp_display_results.get("is_left_hand_off"):
                cv2.putText(image, "Left Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            if mp_display_results.get("is_right_hand_off"):
                cv2.putText(image, "Right Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing

            # 얼굴 가림 감지
            # if mp_display_results.get("is_eye_occluded_danger"):
            #     cv2.putText(image, "Eyes Occluded!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            #     text_y_offset += text_spacing
            # elif mp_display_results.get("is_mouth_occluded_as_yawn"):
            #     cv2.putText(image, "Mouth Occluded (Yawn?)", (text_x_offset, text_y_offset), font, font_scale, color, 2)
            #     text_y_offset += text_spacing
            # elif mp_display_results.get("is_face_occluded_by_hand"):
            #     cv2.putText(image, "Face Occluded by Hand!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            #     text_y_offset += text_spacing
                
            # 기타 유용한 정보 표시 (선택 사항)
            # cv2.putText(image, f"EAR: {mp_display_results['mp_ear']:.2f}", (w - 150, 30), font, 0.7, (255, 255, 0), 1)
            # cv2.putText(image, f"MAR: {mp_display_results['mp_mar']:.2f}", (w - 150, 60), font, 0.7, (255, 255, 0), 1)
            
            if mp_display_results.get("is_head_down"):
                # config.json에서 true_pitch_threshold 값 가져오기 (코-눈 거리 기반)
                from config_manager import get_mediapipe_config
                true_pitch_threshold = get_mediapipe_config("true_pitch_threshold", 10.0)
                cv2.putText(image, f"Head Down! (true_pitch_thresh: {true_pitch_threshold:.1f})", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            elif mp_display_results.get("is_head_up"):
                cv2.putText(image, "Head Up!", (text_x_offset, text_y_offset), font, font_scale, (0, 165, 255), 2)  # 주황색
                text_y_offset += text_spacing

            # ⭐ 실제 감지에 사용되는 EAR 값과 임계값 표시
            mp_ear = mp_display_results.get("mp_ear", None)
            eye_blink_threshold = mp_display_results.get("eye_blink_threshold", None)
            if mp_ear is not None and eye_blink_threshold is not None:
                ear_color = (0, 255, 0) if mp_ear < eye_blink_threshold else warning_color
                cv2.putText(image, f"EAR: {mp_ear:.3f} (thresh: {eye_blink_threshold:.3f})", (text_x_offset, text_y_offset), font, font_scale, ear_color, 2)
                text_y_offset += text_spacing

            # ⭐ YAWN 값 표시 (EAR 바로 아래)
            mar_value = mp_display_results.get("mar_value", None)
            if mar_value is not None:
                # config.json에서 jaw_open_threshold 값 가져오기
                from config_manager import get_mediapipe_config
                jaw_open_threshold = get_mediapipe_config("jaw_open_threshold", 0.4)
                yawn_color = (0, 255, 255) if mar_value > jaw_open_threshold else (0, 255, 0)  # 노란색 if yawning, 초록색 if normal
                cv2.putText(image, f"YAWN: {mar_value:.3f} (thresh: {jaw_open_threshold:.3f})", (text_x_offset, text_y_offset), font, font_scale, yawn_color, 2)
                text_y_offset += text_spacing

        else:
            # 캘리브레이션되지 않은 경우 표시
            cv2.putText(image, "Not Calibrated", (text_x_offset, text_y_offset), font, font_scale, (128, 128, 128), 2)
            text_y_offset += text_spacing
        
        return image

    def draw_fps(self, frame, fps):
        h, w = frame.shape[:2]
        # 중앙 위에 FPS 표시 - crop offset 적용
        fps_text = f"FPS: {fps:.2f}"
        text_size = cv2.getTextSize(fps_text, self.font, self.font_scale, self.thickness)[0]
        text_x = (w - text_size[0]) // 2  # 중앙 정렬
        cv2.putText(frame, fps_text, (text_x, 30 + self.crop_offset), self.font, self.font_scale, (0, 255, 255), self.thickness, cv2.LINE_AA)
        return frame

    def draw_mediapipe_front_status(self, image, is_calibrated, is_distracted):
        status_text = f"MP Calibrated: {'Yes' if is_calibrated else 'No'}"
        status_color = (0, 255, 0) if is_calibrated else (0, 0, 255)
        cv2.putText(image, status_text, (10, 180 + self.crop_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if is_calibrated:
            distracted_text = f"MP Distracted: {'Yes' if is_distracted else 'No'}"
            distracted_color = (0, 0, 255) if is_distracted else (0, 255, 0)
            cv2.putText(image, distracted_text, (10, 210 + self.crop_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distracted_color, 2)
        return image

    def draw_mediapipe_roi(self, image, roi_bounds, is_calibrated=False, is_face_in_roi=True):
        """
        MediaPipe ROI를 시각화합니다.
        roi_bounds: (x1, y1, x2, y2) 형태의 ROI 경계
        is_calibrated: 캘리브레이션 완료 여부
        is_face_in_roi: ROI 내에 얼굴이 있는지 여부
        """
        if roi_bounds is None:
            return image
        
        x1, y1, x2, y2 = roi_bounds
        
        # 캘리브레이션 상태에 따른 색상 설정
        if not is_calibrated:
            # 캘리브레이션 전: 회색 점선
            color = (128, 128, 128)
            thickness = 2
            line_type = cv2.LINE_8
        else:
            # 캘리브레이션 후: 얼굴이 ROI 내에 있으면 초록색, 없으면 빨간색
            if is_face_in_roi:
                color = (0, 255, 0)  # 초록색
                thickness = 3
                line_type = cv2.LINE_8
            else:
                color = (0, 0, 255)  # 빨간색
                thickness = 3
                line_type = cv2.LINE_8
        
        # ROI 사각형 그리기
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, line_type)
        
        # ROI 라벨 추가
        label = "MediaPipe ROI"
        if not is_calibrated:
            label += " (Not Calibrated)"
        elif not is_face_in_roi:
            label += " (Face Outside)"
        else:
            label += " (Calibrated)"
        
        # 라벨 배경
        label_size = cv2.getTextSize(label, self.font, 0.6, 2)[0]
        cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                     (int(x1) + label_size[0], int(y1)), color, -1)
        
        # 라벨 텍스트
        cv2.putText(image, label, (int(x1), int(y1) - 5), 
                   self.font, 0.6, (255, 255, 255), 2)
        
        return image
