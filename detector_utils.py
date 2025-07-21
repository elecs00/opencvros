from scipy.spatial import distance as dist
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation
from collections import deque

# 기존 EAR/MAR 함수 (3DDFA/dlib 공용)
def calculate_ear(eye):
    """눈의 EAR 계산 (디버그용)"""
    print(f"[DEBUG] EAR input points: {eye}")
    if len(eye) < 6:
        return 0.0
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    """20점 입력 (dlib 68점 기준 48~67)"""
    # dlib 68점 기준: mouth[0]~mouth[19] == 48~67
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (2.0 * D)

def calculate_mar_35(landmarks):
    """35점 랜드마크용 MAR 계산 (입 중앙/좌우 고정 인덱스 사용, MAR이 작을수록 입을 다문 상태)"""
    if len(landmarks) < 12:
        return 0.0
    # 8: 윗입술 중앙, 9: 아랫입술 중앙, 10: 왼쪽 입꼬리, 11: 오른쪽 입꼬리
    top_lip = landmarks[8]
    bottom_lip = landmarks[9]
    left_corner = landmarks[10]
    right_corner = landmarks[11]
    mouth_height = dist.euclidean(top_lip, bottom_lip)
    mouth_width = dist.euclidean(left_corner, right_corner)
    if mouth_height < 1e-6:
        return 0.0
    mar = mouth_width / mouth_height  # 식을 반전: 작을수록 입을 다문 상태
    return mar


# dlib 68점 기준 head pose 계산 함수
def get_head_pose(shape, size):
    """
    shape: (68, 2) ndarray (landmarks)
    size: (h, w) tuple (frame size)
    return: (pitch, yaw, roll, origin_point2D, nose_end_point2D)
    """
    # 3D 모델 포인트 (dlib 68점 기준)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (30)
        (0.0, -330.0, -65.0),        # Chin (8)
        (-225.0, 170.0, -135.0),     # Left eye left corner (36)
        (225.0, 170.0, -135.0),      # Right eye right corner (45)
        (-150.0, -150.0, -125.0),    # Left mouth corner (48)
        (150.0, -150.0, -125.0)      # Right mouth corner (54)
    ], dtype=np.float32)
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")
    h, w = size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    # nose 방향 벡터 (z축)
    nose_end_3D = np.array([[0, 0, 100.0]], dtype=np.float32)
    nose_start_3D = np.array([[0, 0, 0]], dtype=np.float32)
    nose_end_2D, _ = cv2.projectPoints(nose_end_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    nose_start_2D, _ = cv2.projectPoints(nose_start_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    # 오일러 각도 변환
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    return pitch, yaw, roll, tuple(nose_start_2D[0][0]), tuple(nose_end_2D[0][0])

def get_head_pose_35(landmarks, size):
    """
    35개 랜드마크용 head pose 계산 (정확한 버전)
    landmarks: 35개 랜드마크 리스트
    size: (h, w) tuple (frame size)
    return: (pitch, yaw, roll, origin_point2D, nose_end_point2D)
    """
    def get_xy(pt):
        # pt가 (idx, (x, y))면 (x, y) 반환, (x, y)면 그대로 반환
        if isinstance(pt, tuple) and len(pt) == 2 and isinstance(pt[1], tuple):
            return pt[1]
        return pt
    if len(landmarks) < 35:
        return 0.0, 0.0, 0.0, (0, 0), (0, 0)
    h, w = size
    try:
        # 수정 (인덱스 직접 지정)
        nose_tip = landmarks[4]      # 또는 7, 실제로 더 nose tip에 가까운 쪽 선택
        chin = landmarks[26]
        left_eye_corner = landmarks[1]
        right_eye_corner = landmarks[3]
        left_mouth_corner = landmarks[8]
        right_mouth_corner = landmarks[9]

        image_points = np.array([
            nose_tip,
            chin,
            left_eye_corner,
            right_eye_corner,
            left_mouth_corner,
            right_mouth_corner
        ], dtype="double")
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0, (0, 0), (0, 0)

        # --- Scipy를 이용한 안정적인 각도 계산 (ZYX 순서) ---
        r = Rotation.from_rotvec(rotation_vector.flatten())
        # ZYX 순서는 Roll, Yaw, Pitch 순으로 해석됨
        angles = r.as_euler('zyx', degrees=True)
        
        roll = angles[0]
        yaw = angles[1]
        pitch = angles[2]
        
        # 직관적인 방향으로 조정
        # Pitch(X축 회전): OpenCV에서 +X 회전은 위를 향함. 아래를 +로 하기 위해 부호 변경.
        pitch = -pitch
        # Yaw(Y축 회전): OpenCV에서 +Y 회전은 왼쪽을 향함. 오른쪽을 +로 하기 위해 부호 변경.
        yaw = -yaw
        # Roll(Z축 회전): OpenCV에서 +Z 회전은 왼쪽으로 기울어짐. 오른쪽을 +로 하기 위해 부호 변경.
        roll = -roll

        # 각도 정규화 (-180 ~ 180도 범위로 제한)
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            return angle
        
        pitch = normalize_angle(pitch)
        yaw = normalize_angle(yaw)
        roll = normalize_angle(roll)
        
        # 디버그 출력 (각도 변화 추적)
        print(f"[detector_utils] Raw angles - P:{pitch:.1f}° Y:{yaw:.1f}° R:{roll:.1f}°")
        
        # 추가 안정화: 급격한 각도 변화 방지
        # 이전 프레임과의 차이가 180도 이상이면 정규화
        if hasattr(get_head_pose_35, 'prev_pitch'):
            pitch_diff = abs(pitch - get_head_pose_35.prev_pitch)
            if pitch_diff > 180:
                if pitch > get_head_pose_35.prev_pitch:
                    pitch -= 360
                else:
                    pitch += 360
            pitch = normalize_angle(pitch)
        get_head_pose_35.prev_pitch = pitch
        
        print(f"[detector_utils] Final angles - P:{pitch:.1f}° Y:{yaw:.1f}° R:{roll:.1f}°")

        # 코 방향 벡터 계산 (시각화용)
        nose_end_2D, _ = cv2.projectPoints(np.array([[0.0, 0.0, 500.0]]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_2D[0][0][0]), int(nose_end_2D[0][0][1]))

        return pitch, yaw, roll, p1, p2
        
    except Exception as e:
        print(f"[detector_utils] Head pose calculation failed: {e}")
        return 0.0, 0.0, 0.0, (0, 0), (0, 0)

def visualize_landmarks(image, landmarks):
    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(image, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

def update_histories_5pt(history_dict, landmarks):
    """5점 랜드마크용 히스토리 업데이트 (눈/입)"""
    if len(landmarks) >= 3:
        history_dict['left_eye_x'].append(landmarks[0][0])
        history_dict['left_eye_y'].append(landmarks[0][1])
        history_dict['right_eye_x'].append(landmarks[1][0])
        history_dict['right_eye_y'].append(landmarks[1][1])
        history_dict['mouth_x'].append(landmarks[2][0])
        history_dict['mouth_y'].append(landmarks[2][1])

def is_eye_closed_5pt(history_dict, jump_thresh=6.0, var_thresh=4.0, mouth_jump_thresh=6.0, mouth_var_thresh=4.0, gaze_thresh=0.3, look_ahead=None):
    """5점 랜드마크 jump/var 기반 눈 감음 인식 (눈동자 움직임 제외)"""
    if len(history_dict['left_eye_x']) < 10:
        return False
    
    left_x = np.array(history_dict['left_eye_x'])
    right_x = np.array(history_dict['right_eye_x'])
    mouth_x = np.array(history_dict['mouth_x'])
    mouth_y = np.array(history_dict['mouth_y'])
    
    # 기존 jump/var 계산
    left_x_jump = left_x.max() - left_x.min()
    right_x_jump = right_x.max() - right_x.min()
    left_x_var = left_x.var()
    right_x_var = right_x.var()
    mouth_x_jump = mouth_x.max() - mouth_x.min()
    mouth_y_jump = mouth_y.max() - mouth_y.min()
    mouth_x_var = mouth_x.var()
    mouth_y_var = mouth_y.var()
    
    # 눈동자 움직임 감지 (양쪽 눈이 같은 방향으로 움직이는지)
    def detect_synchronized_gaze_movement():
        """양쪽 눈이 같은 방향으로 움직이는지 감지"""
        if len(left_x) < 5 or len(right_x) < 5:
            return False
        
        # 최근 5프레임의 움직임 방향 계산
        left_direction = np.diff(left_x[-5:])  # 연속 프레임 간 차이
        right_direction = np.diff(right_x[-5:])
        
        # 같은 방향으로 움직이는지 확인 (상관계수 사용)
        if len(left_direction) > 1 and len(right_direction) > 1:
            correlation = np.corrcoef(left_direction, right_direction)[0, 1]
            # 상관계수가 높으면 같은 방향으로 움직임 (시선 이동)
            return not np.isnan(correlation) and correlation > 0.7
        
        return False
    
    # 눈동자가 같은 방향으로 움직이는지 확인
    synchronized_gaze = detect_synchronized_gaze_movement()
    
    print(f"[DEBUG] Eye movement - Left jump: {left_x_jump:.2f}, Right jump: {right_x_jump:.2f}")
    print(f"[DEBUG] Eye variance - Left var: {left_x_var:.2f}, Right var: {right_x_var:.2f}")
    print(f"[DEBUG] Synchronized gaze movement: {synchronized_gaze}")
    if look_ahead is not None:
        print(f"[DEBUG] Look ahead status: {look_ahead}")
    
    # 기준값(실험적으로 조정)
    jump_thresh = 6.0
    var_thresh = 4.0
    mouth_jump_thresh = 6.0
    mouth_var_thresh = 4.0
    
    # 눈은 jump/var가 크고, 입은 jump/var가 작으면 눈 감음
    # 단, 눈동자가 같은 방향으로 움직이면 시선 이동으로 간주하여 눈 감음이 아님
    if ((left_x_jump > jump_thresh or right_x_jump > jump_thresh or
         left_x_var > var_thresh or right_x_var > var_thresh)):
        # 입도 같이 크면 고개 움직임으로 간주(눈 감음 아님)
        if (mouth_x_jump > mouth_jump_thresh or mouth_y_jump > mouth_jump_thresh or
            mouth_x_var > mouth_var_thresh or mouth_y_var > mouth_var_thresh):
            print(f"[DEBUG] Eye closed detection: False (head movement)")
            return False
        # 눈동자가 같은 방향으로 움직이면 시선 이동으로 간주(눈 감음 아님)
        if synchronized_gaze:
            print(f"[DEBUG] Eye closed detection: False (synchronized gaze movement)")
            return False
        print(f"[DEBUG] Eye closed detection: True")
        return True
    
    print(f"[DEBUG] Eye closed detection: False (no significant movement)")
    return False

def calculate_head_down_by_mouth_chin_distance(landmarks, head_down_threshold=0.11, head_up_threshold=0.22):
    """
    35점 랜드마크에서 입-턱 거리로 고개 숙임/들기 감지
    landmarks: 35개 랜드마크 리스트
    head_down_threshold: 고개 숙임 임계값 (기본값 0.11)
    head_up_threshold: 고개 들기 임계값 (기본값 0.22)
    return: (is_head_down, is_head_up, distance, normalized_distance)
    """
    if len(landmarks) < 35:
        return False, False, 0.0, 0.0
    
    try:
        # 입 오른쪽 점 (오른쪽 입꼬리) - 11번 인덱스
        mouth_right = landmarks[11]
        
        # 턱점 - 26번 인덱스
        chin = landmarks[26]
        
        # 입-턱 거리 계산
        distance = dist.euclidean(mouth_right, chin)
        
        # 얼굴 크기로 정규화 (얼굴 대각선 길이 사용)
        face_width = max(p[0] for p in landmarks) - min(p[0] for p in landmarks)
        face_height = max(p[1] for p in landmarks) - min(p[1] for p in landmarks)
        face_diagonal = np.sqrt(face_width**2 + face_height**2)
        
        normalized_distance = distance / face_diagonal if face_diagonal > 0 else 0.0
        
        # 고개 숙임/들기 판정
        is_head_down = normalized_distance <= head_down_threshold
        is_head_up = normalized_distance >= head_up_threshold
        
        return is_head_down, is_head_up, distance, normalized_distance
        
    except Exception as e:
        print(f"[detector_utils] Head down calculation failed: {e}")
        return False, False, 0.0, 0.0

def normalize_angle(angle):
    """각도를 -180 ~ 180도 범위로 정규화"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def calculate_pupil_gaze_35(landmarks_35, landmarks_5=None, calibrated_landmarks_35=None, calibrated_landmarks_5=None, calibrated_gaze_ratio=None):
    """
    35점 랜드마크와 5점 랜드마크를 조합하여 눈동자 시선 방향 계산
    캘리브레이션 시 가상의 정면 점을 설정하고, 고개를 돌려도 그 점을 바라보면 정면 응시로 판단
    landmarks_35: 현재 35개 랜드마크 (눈 크기/방향용)
    landmarks_5: 현재 5개 랜드마크 (정확한 눈동자 위치용)
    calibrated_landmarks_35: 캘리브레이션된 35개 랜드마크
    calibrated_landmarks_5: 캘리브레이션된 5개 랜드마크
    calibrated_gaze_ratio: 캘리브레이션 시 눈동자의 상대적 위치 비율
    return: (left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y, is_looking_ahead)
    """
    if len(landmarks_35) < 35 or not landmarks_5 or len(landmarks_5) < 2:
        return 0.0, 0.0, 0.0, 0.0, False
    
    try:
        # 5점 모델에서 정확한 눈동자 위치 가져오기
        left_pupil = landmarks_5[0]   # 왼쪽 눈동자 (5점 모델의 첫 번째 점)
        right_pupil = landmarks_5[1]  # 오른쪽 눈동자 (5점 모델의 두 번째 점)
        
        # 35점 모델에서 눈 주변 점들로 눈의 크기와 중심 계산
        # 35점 모델의 눈 관련 인덱스: 왼쪽 눈(0,1), 오른쪽 눈(2,3)
        left_eye_points = [landmarks_35[0], landmarks_35[1]]  # 왼쪽 눈 안쪽(0), 바깥쪽(1)
        right_eye_points = [landmarks_35[2], landmarks_35[3]]  # 오른쪽 눈 안쪽(2), 바깥쪽(3)
        
        def calculate_eye_gaze_ratio(pupil, eye_points, is_left_eye=True):
            """눈동자가 눈의 어느 위치에 있는지 비율로 계산 (0~1 범위)"""
            if len(eye_points) < 2:
                return 0.5, 0.5  # 기본값: 중앙
            
            # 35점 랜드마크에서 눈 주변의 더 많은 점들을 사용
            if is_left_eye:
                # 왼쪽 눈: 0,1번 (안쪽, 바깥쪽) + 추가 점들
                eye_region_points = [landmarks_35[0], landmarks_35[1]]
                # 왼쪽 눈 주변 추가 점들 (실제 35점 모델의 인덱스에 맞게 조정 필요)
                # 예: 눈썹, 눈꼬리 등 추가 점들
            else:
                # 오른쪽 눈: 2,3번 (안쪽, 바깥쪽) + 추가 점들
                eye_region_points = [landmarks_35[2], landmarks_35[3]]
                # 오른쪽 눈 주변 추가 점들
            
            # 기존 방식: 단순히 두 점의 min/max 사용
            eye_left = min(eye_points[0][0], eye_points[1][0])  # 더 왼쪽
            eye_right = max(eye_points[0][0], eye_points[1][0])  # 더 오른쪽
            eye_top = min(eye_points[0][1], eye_points[1][1])    # 더 위쪽
            eye_bottom = max(eye_points[0][1], eye_points[1][1]) # 더 아래쪽
            
            # 개선된 방식: 눈의 실제 크기를 더 정확하게 계산
            # 눈의 가로 크기를 약간 확장 (눈동자가 경계에 있을 때를 대비)
            eye_width = eye_right - eye_left
            eye_height = eye_bottom - eye_top
            
            # 눈동자가 경계에 있을 때를 대비하여 약간의 여유 공간 추가
            margin_x = eye_width * 0.1  # 10% 여유
            margin_y = eye_height * 0.1  # 10% 여유
            
            eye_left_expanded = eye_left - margin_x
            eye_right_expanded = eye_right + margin_x
            eye_top_expanded = eye_top - margin_y
            eye_bottom_expanded = eye_bottom + margin_y
            
            # 눈동자의 상대적 위치 (0~1 범위)
            if eye_right_expanded - eye_left_expanded > 0:
                gaze_x_ratio = (pupil[0] - eye_left_expanded) / (eye_right_expanded - eye_left_expanded)
            else:
                gaze_x_ratio = 0.5
                
            if eye_bottom_expanded - eye_top_expanded > 0:
                gaze_y_ratio = (pupil[1] - eye_top_expanded) / (eye_bottom_expanded - eye_top_expanded)
            else:
                gaze_y_ratio = 0.5
            
            # 범위 제한
            gaze_x_ratio = np.clip(gaze_x_ratio, 0.0, 1.0)
            gaze_y_ratio = np.clip(gaze_y_ratio, 0.0, 1.0)
            
            return gaze_x_ratio, gaze_y_ratio
        
        # 현재 눈동자 위치 비율 계산
        left_gaze_x_ratio, left_gaze_y_ratio = calculate_eye_gaze_ratio(left_pupil, left_eye_points, is_left_eye=True)
        right_gaze_x_ratio, right_gaze_y_ratio = calculate_eye_gaze_ratio(right_pupil, right_eye_points, is_left_eye=False)
        
        print(f"[DEBUG] Current gaze ratios - Left: ({left_gaze_x_ratio:.3f}, {left_gaze_y_ratio:.3f}), Right: ({right_gaze_x_ratio:.3f}, {right_gaze_y_ratio:.3f})")
        
        # 캘리브레이션된 값이 있으면 상대적 시선 방향 계산
        if (calibrated_landmarks_35 and len(calibrated_landmarks_35) >= 35 and 
            calibrated_landmarks_5 and len(calibrated_landmarks_5) >= 2):
            
            cal_left_pupil = calibrated_landmarks_5[0]
            cal_right_pupil = calibrated_landmarks_5[1]
            cal_left_eye_points = [calibrated_landmarks_35[0], calibrated_landmarks_35[1]]
            cal_right_eye_points = [calibrated_landmarks_35[2], calibrated_landmarks_35[3]]
            
            # 캘리브레이션 시 눈동자 위치 비율 계산
            cal_left_gaze_x_ratio, cal_left_gaze_y_ratio = calculate_eye_gaze_ratio(cal_left_pupil, cal_left_eye_points, is_left_eye=True)
            cal_right_gaze_x_ratio, cal_right_gaze_y_ratio = calculate_eye_gaze_ratio(cal_right_pupil, cal_right_eye_points, is_left_eye=False)
            
            print(f"[DEBUG] Calibrated gaze ratios - Left: ({cal_left_gaze_x_ratio:.3f}, {cal_left_gaze_y_ratio:.3f}), Right: ({cal_right_gaze_x_ratio:.3f}, {cal_right_gaze_y_ratio:.3f})")
            
            # 상대적 시선 방향 (캘리브레이션 기준 대비)
            rel_left_gaze_x = left_gaze_x_ratio - cal_left_gaze_x_ratio
            rel_left_gaze_y = left_gaze_y_ratio - cal_left_gaze_y_ratio
            rel_right_gaze_x = right_gaze_x_ratio - cal_right_gaze_x_ratio
            rel_right_gaze_y = right_gaze_y_ratio - cal_right_gaze_y_ratio
            
            print(f"[DEBUG] Relative gaze - Left: ({rel_left_gaze_x:.3f}, {rel_left_gaze_y:.3f}), Right: ({rel_right_gaze_x:.3f}, {rel_right_gaze_y:.3f})")
            
            # 상대값 그대로 사용 (정면이면 0에 가까움)
            left_gaze_x = rel_left_gaze_x
            left_gaze_y = rel_left_gaze_y
            right_gaze_x = rel_right_gaze_x
            right_gaze_y = rel_right_gaze_y
            
            print(f"[DEBUG] Final gaze - Left: ({left_gaze_x:.3f}, {left_gaze_y:.3f}), Right: ({right_gaze_x:.3f}, {right_gaze_y:.3f})")
            
            return left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y, False
        
        # 캘리브레이션 없으면 절대적 위치 사용
        left_gaze_x = left_gaze_x_ratio * 2 - 1  # 0~1을 -1~1로 변환
        left_gaze_y = left_gaze_y_ratio * 2 - 1
        right_gaze_x = right_gaze_x_ratio * 2 - 1
        right_gaze_y = right_gaze_y_ratio * 2 - 1
        
        return left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y, False
        
    except Exception as e:
        print(f"[detector_utils] Pupil gaze calculation failed: {e}")
        return 0.0, 0.0, 0.0, 0.0, False

def is_looking_ahead_35(landmarks_35, landmarks_5=None, calibrated_landmarks_35=None, calibrated_landmarks_5=None, gaze_threshold=1.0, head_rotation_threshold=30.0, head_pose=None):
    """
    35점 랜드마크와 5점 랜드마크를 조합하여 정면 응시 여부 판단 및 시선 방향 문자열 반환
    gaze_threshold: 정면 기준(1.0), 좌(0.8), 우(1.2) 등 커스텀
    head_rotation_threshold: 고개 회전 임계값(30도)
    return: (상태문자열, is_looking_ahead, gaze_info)
    """
    if len(landmarks_35) < 35 or not landmarks_5 or len(landmarks_5) < 2:
        return "Gaze: OFF", False, {}
    try:
        left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y, _ = calculate_pupil_gaze_35(
            landmarks_35, landmarks_5, calibrated_landmarks_35, calibrated_landmarks_5
        )
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
        avg_gaze_y = (left_gaze_y + right_gaze_y) / 2
        gaze_magnitude = np.sqrt(avg_gaze_x**2 + avg_gaze_y**2)
        
        # 캘리브레이션 여부 확인
        is_calibrated = (calibrated_landmarks_35 is not None and len(calibrated_landmarks_35) >= 35 and 
                        calibrated_landmarks_5 is not None and len(calibrated_landmarks_5) >= 2)
        
        print(f"[DEBUG] is_looking_ahead_35 - is_calibrated: {is_calibrated}")
        
        # 고개 회전 체크 (YAW값만 사용, 캘리브레이션 여부에 따라 임계값 조정)
        if head_pose:
            try:
                pitch, yaw, roll = head_pose
                if yaw is not None:
                    if is_calibrated:
                        # 캘리브레이션 후: 더 큰 YAW 임계값 사용 (60도)
                        if abs(float(yaw)) > head_rotation_threshold * 2.0:
                            print(f"[DEBUG] is_looking_ahead_35 - Head YAW rotation too large (calibrated): Y={yaw:.1f}°")
                            return "Gaze: OFF", False, {"gaze_magnitude": gaze_magnitude}
                    else:
                        # 캘리브레이션 안된 경우: 기본 YAW 임계값 사용 (30도)
                        if abs(float(yaw)) > head_rotation_threshold:
                            print(f"[DEBUG] is_looking_ahead_35 - Head YAW rotation too large (not calibrated): Y={yaw:.1f}°")
                            return "Gaze: OFF", False, {"gaze_magnitude": gaze_magnitude}
            except Exception:
                pass  # head_pose 값이 이상하면 GAZE OFF 조건 무시
        
        # 시선 방향 판정
        print(f"[DEBUG] is_looking_ahead_35 - gaze_magnitude: {gaze_magnitude:.6f}, gaze_threshold: {gaze_threshold}")
        if gaze_magnitude <= gaze_threshold:
            status = "Look A head"
            is_looking_ahead = True
            print(f"[DEBUG] is_looking_ahead_35 - Condition TRUE: gaze_magnitude({gaze_magnitude:.6f}) <= gaze_threshold({gaze_threshold})")
        else:
            status = "Look Away"
            is_looking_ahead = False
            print(f"[DEBUG] is_looking_ahead_35 - Condition FALSE: gaze_magnitude({gaze_magnitude:.6f}) > gaze_threshold({gaze_threshold})")
        print(f"[DEBUG] is_looking_ahead_35 - Final result: status='{status}', is_looking_ahead={is_looking_ahead}")
        gaze_info = {
            'left_gaze_x': left_gaze_x,
            'left_gaze_y': left_gaze_y,
            'right_gaze_x': right_gaze_x,
            'right_gaze_y': right_gaze_y,
            'avg_gaze_x': avg_gaze_x,
            'avg_gaze_y': avg_gaze_y,
            'gaze_magnitude': gaze_magnitude
        }
        return status, is_looking_ahead, gaze_info
    except Exception as e:
        print(f"[detector_utils] Looking ahead detection failed: {e}")
        return "Gaze: OFF", False, {} 

# ============================================================================
# MediaPipe 관련 계산 함수들
# ============================================================================

def get_mediapipe_head_pose_from_landmarks(frame_size, face_landmarks):
    """
    MediaPipe Face Mesh에서 head pose 계산 (기존 방식과 호환)
    
    Args:
        frame_size: (height, width) 튜플
        face_landmarks: MediaPipe Face Mesh 랜드마크 (Task API 또는 Face Mesh 방식)
    
    Returns:
        dict: head pose 정보 또는 None (실패 시)
    """
    try:
        # Task API와 기존 방식 모두 지원하도록 랜드마크 처리
        if hasattr(face_landmarks, 'landmark'):
            # 기존 Face Mesh 방식
            landmarks = np.array([[lm.x * frame_size[1], lm.y * frame_size[0], lm.z] for lm in face_landmarks.landmark])
        else:
            # Task API 방식 - 리스트 형태
            landmarks = np.array([[lm.x * frame_size[1], lm.y * frame_size[0], lm.z] for lm in face_landmarks])
        
        # 필요한 랜드마크 인덱스들이 존재하는지 확인
        required_indices = [1, 152, 33, 263, 61, 291]  # nose, chin, left_eye, right_eye, left_mouth, right_mouth
        if len(landmarks) < max(required_indices) + 1:
            print(f"[detector_utils] Not enough landmarks: {len(landmarks)} < {max(required_indices) + 1}")
            return None
        
        # 얼굴의 주요 포인트들
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # 얼굴 중심점
        face_center = np.mean([left_eye, right_eye, left_mouth, right_mouth], axis=0)
        
        # 카메라 내부 파라미터 (대략적인 값)
        focal_length = frame_size[1]
        center = (frame_size[1] / 2, frame_size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1))
        
        # 3D 모델 포인트들 (얼굴의 3D 좌표)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 2D 이미지 포인트들 (x, y 좌표만)
        image_points = np.array([
            [landmarks[1][0], landmarks[1][1]],    # Nose tip
            [landmarks[152][0], landmarks[152][1]],  # Chin
            [landmarks[33][0], landmarks[33][1]],   # Left eye left corner
            [landmarks[263][0], landmarks[263][1]],  # Right eye right corner
            [landmarks[61][0], landmarks[61][1]],   # Left mouth corner
            [landmarks[291][0], landmarks[291][1]]   # Right mouth corner
        ], dtype=np.float64)
        
        # 포인트 개수와 형식 검증
        if image_points.shape[0] < 4:
            print(f"[detector_utils] Not enough image points: {image_points.shape[0]} < 4")
            return None
        
        # PnP 문제 해결
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # 회전 벡터를 회전 행렬로 변환
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Euler 각도 계산
            pitch_val = np.arctan2(-rotation_mat[2, 0], np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2)) * 180 / np.pi
            yaw_val = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]) * 180 / np.pi
            roll_val = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]) * 180 / np.pi

            # 사용자 피드백 기반으로 yaw와 pitch를 스왑
            # 고개 숙이기(pitch)가 좌우 회전(yaw)으로, 좌우 회전이 고개 숙이기로 계산되는 문제 수정
            pitch = yaw_val
            yaw = pitch_val
            roll = roll_val
            
            # Roll 각도를 -180~180도 범위로 정규화
            if roll > 180:
                roll -= 360
            elif roll < -180:
                roll += 360
            
            return {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "rotation_mat": rotation_mat,
                "translation_vec": translation_vec
            }
    except Exception as e:
        print(f"[detector_utils] MediaPipe head pose calculation error: {e}")
    
    return None

def get_mediapipe_head_pose_from_matrix(transformation_matrix):
    """
    Task API의 transformation_matrix에서 head pose 계산
    
    Args:
        transformation_matrix: MediaPipe Task API의 facial transformation matrix
    
    Returns:
        tuple: (pitch, yaw, roll) 각도 (도 단위)
    """
    # 회전 행렬 추출
    rotation_matrix = transformation_matrix[0:3, 0:3]
    
    try:
        # Euler 각도 계산
        yaw_val = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)) * 180 / np.pi * 2
        roll_val = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi * 2
        pitch_val = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
        
        # 사용자 피드백 기반으로 yaw와 pitch를 스왑
        # 고개 숙이기(pitch)가 좌우 회전(yaw)으로, 좌우 회전이 고개 숙이기로 계산되는 문제 수정
        pitch = pitch_val
        yaw = yaw_val
        roll = roll_val
        
        # Roll 각도를 -180~180도 범위로 정규화
        if roll > 180:
            roll -= 360
        elif roll < -180:
            roll += 360

        return pitch, yaw, roll
    except Exception as e:
        print(f"[detector_utils] Error calculating head pose from matrix: {e}")
        return 0, 0, 0

def get_mediapipe_true_pitch_from_landmarks(landmarks):
    """
    MediaPipe 랜드마크에서 true pitch 계산 (이마-턱 벡터 기반)
    
    Args:
        landmarks: MediaPipe 랜드마크 (Task API는 리스트, FaceMesh는 .landmark)
    
    Returns:
        float: true pitch 각도 (도 단위)
    """
    if hasattr(landmarks, 'landmark'):
        lm = landmarks.landmark
    else:
        lm = landmarks
    
    try:
        # 이마와 턱을 연결하는 벡터로 고개 숙임 계산
        # 이마: 랜드마크 10번 (이마 중앙)
        # 턱: 랜드마크 152번 (턱 중앙)
        forehead = np.array([lm[10].x, lm[10].y, lm[10].z])
        chin = np.array([lm[152].x, lm[152].y, lm[152].z])
        
        # 이마-턱 벡터 계산
        face_vec = chin - forehead
        face_vec_normalized = face_vec / np.linalg.norm(face_vec)
        
        # 수직 벡터 (y축)와의 각도 계산
        vertical_vec = np.array([0, 1, 0])  # y축 (위쪽 방향)
        
        # 두 벡터 사이의 각도 계산
        dot_product = np.clip(np.dot(face_vec_normalized, vertical_vec), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # 고개 숙임/들기 방향 결정
        # face_vec의 y성분이 양수면 고개를 숙인 것, 음수면 고개를 든 것
        if face_vec[1] > 0:
            pitch_deg = angle_deg  # 고개 숙임 (양수)
        else:
            pitch_deg = -angle_deg  # 고개 들기 (음수)
        
        return pitch_deg
    except Exception as e:
        print(f"[detector_utils] True pitch calculation failed: {e}")
        return 0.0

def get_mediapipe_pupil_center(face_landmarks):
    """
    MediaPipe 랜드마크에서 양쪽 눈동자의 중심점 계산
    
    Args:
        face_landmarks: MediaPipe Face Mesh 랜드마크
    
    Returns:
        tuple: (x, y) 정규화된 좌표 또는 None (실패 시)
    """
    try:
        if hasattr(face_landmarks, 'landmark'):
            # 기존 Face Mesh 방식
            landmarks = face_landmarks.landmark
            # MediaPipe Face Mesh의 눈동자 랜드마크 인덱스
            left_pupil = np.array([landmarks[468].x, landmarks[468].y])   # 왼쪽 눈동자 중심
            right_pupil = np.array([landmarks[473].x, landmarks[473].y])  # 오른쪽 눈동자 중심
        else:
            # Task API 방식 - 리스트 형태
            landmarks = face_landmarks
            left_pupil = np.array([landmarks[468].x, landmarks[468].y])
            right_pupil = np.array([landmarks[473].x, landmarks[473].y])
        
        # 양쪽 눈동자의 중심점 계산
        pupil_center = (left_pupil + right_pupil) / 2
        return pupil_center
        
    except Exception as e:
        print(f"[detector_utils] Error calculating pupil center: {e}")
        return None

def calculate_mediapipe_pupil_gaze_deviation(face_landmarks, frame_size, calibrated_pupil_center, 
                                           pupil_gaze_threshold=0.05, pupil_gaze_consec_frames=10):
    """
    MediaPipe 랜드마크에서 눈동자 기반 시선 이탈 계산
    
    Args:
        face_landmarks: MediaPipe Face Mesh 랜드마크
        frame_size: (height, width) 튜플
        calibrated_pupil_center: 캘리브레이션된 눈동자 중심점
        pupil_gaze_threshold: 시선 이탈 임계값
        pupil_gaze_consec_frames: 연속 프레임 임계값
    
    Returns:
        bool: 시선 이탈 여부
    """
    if calibrated_pupil_center is None:
        return False
    
    current_pupil_center = get_mediapipe_pupil_center(face_landmarks)
    if current_pupil_center is None:
        return False
    
    # 캘리브레이션 시 설정된 가상의 정면 점 (화면 중앙)
    virtual_front_point = np.array([0.5, 0.5])  # 정규화된 좌표 (화면 중앙)
    
    # 현재 눈동자 중심과 가상 정면 점의 거리 계산
    deviation_from_front = np.linalg.norm(current_pupil_center - virtual_front_point)
    
    # 캘리브레이션 시 눈동자가 정면을 바라볼 때의 거리 (기준 거리)
    calibrated_deviation_from_front = np.linalg.norm(calibrated_pupil_center - virtual_front_point)
    
    # 현재 거리와 캘리브레이션 시 거리의 차이
    deviation_diff = abs(deviation_from_front - calibrated_deviation_from_front)
    
    # 얼굴 크기 대비 정규화된 거리 계산
    if hasattr(face_landmarks, 'landmark'):
        landmarks = face_landmarks.landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
    else:
        landmarks = face_landmarks
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
    
    face_width = (max(x_coords) - min(x_coords))
    face_height = (max(y_coords) - min(y_coords))
    face_size = max(face_width, face_height)  # 얼굴 크기 (정규화된 좌표)
    
    # 얼굴 크기 대비 정규화된 거리 차이
    normalized_deviation = deviation_diff / face_size
    
    # 연속 프레임 카운터는 호출하는 쪽에서 관리
    is_deviated = normalized_deviation > pupil_gaze_threshold
    
    if is_deviated:
        print(f"[detector_utils] Pupil gaze deviated: deviation={normalized_deviation:.3f} > {pupil_gaze_threshold}")
    
    return is_deviated

def calculate_mediapipe_face_center_and_size(face_landmarks, frame_size):
    """
    MediaPipe 랜드마크에서 얼굴 중심점과 크기 계산
    
    Args:
        face_landmarks: MediaPipe Face Mesh 랜드마크
        frame_size: (height, width) 튜플
    
    Returns:
        tuple: (face_center, face_size, face_roi) 또는 None (실패 시)
    """
    try:
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
            landmarks = face_landmarks
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            # 코 위치를 기준점으로 사용 (랜드마크 인덱스 1이 코)
            nose_x = landmarks[1].x * frame_size[1]
            nose_y = landmarks[1].y * frame_size[0]
        
        # 코를 중심으로 하는 얼굴 위치 저장
        face_center = (nose_x, nose_y)
        
        face_width = (max(x_coords) - min(x_coords)) * frame_size[1]
        face_height = (max(y_coords) - min(y_coords)) * frame_size[0]
        face_size = (face_width, face_height)
        
        # Calculate face ROI bounds with some margin
        margin_x = face_width * 0.2
        margin_y = face_height * 0.2
        x1 = max(0, nose_x - face_width/2 - margin_x)
        y1 = max(0, nose_y - face_height/2 - margin_y)
        x2 = min(frame_size[1], nose_x + face_width/2 + margin_x)
        y2 = min(frame_size[0], nose_y + face_height/2 + margin_y)
        face_roi = (x1, y1, x2, y2)
        
        return face_center, face_size, face_roi
        
    except Exception as e:
        print(f"[detector_utils] Error calculating face center and size: {e}")
        return None

def is_mediapipe_face_within_calibrated_bounds(face_landmarks, frame_size, calibrated_face_center, 
                                              calibrated_face_size, face_roi_scale=1.5):
    """
    현재 감지된 얼굴이 캘리브레이션된 운전자의 위치와 크기 범위 내에 있는지 확인
    
    Args:
        face_landmarks: 현재 MediaPipe Face Mesh 랜드마크
        frame_size: (height, width) 튜플
        calibrated_face_center: 캘리브레이션된 얼굴 중심점
        calibrated_face_size: 캘리브레이션된 얼굴 크기
        face_roi_scale: 얼굴 ROI 스케일 팩터
    
    Returns:
        bool: 얼굴이 범위 내에 있으면 True
    """
    if calibrated_face_center is None or calibrated_face_size is None:
        return True  # 캘리브레이션되지 않았으면 모든 얼굴 허용
    
    # 현재 얼굴 중심점과 크기 계산
    current_face_info = calculate_mediapipe_face_center_and_size(face_landmarks, frame_size)
    if current_face_info is None:
        return False
    
    current_face_center, current_face_size, _ = current_face_info
    current_face_center_x, current_face_center_y = current_face_center
    current_face_width, current_face_height = current_face_size
    
    # 캘리브레이션된 값들
    calibrated_center_x, calibrated_center_y = calibrated_face_center
    calibrated_width, calibrated_height = calibrated_face_size
    
    # 새로운 방식: 캘리브레이션된 얼굴 ROI의 설정된 배수로 detection 영역 설정
    roi_width = calibrated_width * face_roi_scale
    roi_height = calibrated_height * face_roi_scale
    
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
        print(f"[detector_utils] Face rejected - outside ROI bounds")
        print(f"[detector_utils] ROI: ({roi_x1:.1f}, {roi_y1:.1f}) to ({roi_x2:.1f}, {roi_y2:.1f})")
        print(f"[detector_utils] Face: ({current_face_x1:.1f}, {current_face_y1:.1f}) to ({current_face_x2:.1f}, {current_face_y2:.1f})")
        return False
    
    return True 