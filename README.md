# opencvros — MediaPipe 운전자 상태 모니터링 (ROS2 연동)

카메라 영상에서 운전자의 **졸음·하품·전방 주시 이탈·핸들 파지 여부**를 실시간으로 판정하고, 결과 화면을 ROS2 토픽으로 내보내는 PyQt5 데스크톱 프로그램입니다.

| 항목 | 내용 |
|---|---|
| 인식 | MediaPipe Face Landmarker / Face Mesh(얼굴 랜드마크), Hand Landmarker(손) |
| 판정 | 눈 감김 비율(EAR) → 졸음, 입 벌림(MAR·jaw open) → 하품, 머리 자세(pitch·yaw·roll)와 동공 위치 → 전방 주시 이탈, 손 위치 → 핸들 파지 |
| 보정 | 운전자별 정면 자세·동공 중심 캘리브레이션, 머리를 숙이거나 들 때 눈 감김 임계값 자동 전환 |
| GUI | PyQt5 (시작·정지, 캘리브레이션, 설정 편집·재적용, 손/주의분산 판정 켜고 끄기) |
| ROS2 | 판정 결과가 그려진 화면을 `gui_image`(`sensor_msgs/Image`)로 발행 → C++ 노드가 구독해 표시 |
| 설정 | `config.json` 에서 임계값·연속 프레임 수 조정 |

### 데이터 흐름

```
카메라 → gui_app.py ─▶ mediapipe_analyzer.py (랜드마크·판정) ─▶ visualizer.py (경고 표시)
                    └▶ result_publisher (Python, /gui_image) ─▶ result_subscriber (C++, 화면 표시)
```

## 📁 파일 구조

```
opencvros/
├── gui_app.py               # 진입점: PyQt5 GUI, 카메라 루프, ROS2 발행
├── mediapipe_analyzer.py    # 얼굴·손 랜드마크 분석, 졸음·하품·주시 이탈·핸들 파지 판정, 캘리브레이션
├── detector_utils.py        # EAR·MAR·머리 자세 계산 함수
├── visualizer.py            # 판정 결과를 영상 위에 표시
├── config_manager.py        # config.json 읽기·쓰기
├── config.json              # 임계값 설정
├── requirements.txt
└── Ros2_ws/src/
    ├── result_publisher/    # Python 노드: GUI 화면 → /gui_image
    └── result_subscriber/   # C++ 노드: /gui_image 구독 → OpenCV 창 표시
```

## 🛠️ 설치

1. 저장소 받기
   ```bash
   git clone https://github.com/elecs00/opencvros.git
   cd opencvros
   ```
2. 파이썬 가상환경 (권장)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. 파이썬 패키지
   ```bash
   pip install -r requirements.txt
   ```
4. (선택) MediaPipe Task 모델 — 없으면 기본 Face Mesh 로 자동 대체됩니다.
   ```bash
   mkdir -p models
   wget -O models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```
5. (선택) ROS2 연동 — ROS2 Humble 설치 후
   ```bash
   cd Ros2_ws
   rosdep install -i --from-path src --rosdistro humble -y
   colcon build
   source install/setup.bash
   cd ..
   ```

## 🚀 실행

```bash
# 터미널 1: GUI (ROS2 를 source 한 상태면 /gui_image 도 발행)
python3 gui_app.py

# 터미널 2: (선택) C++ 구독 노드로 발행 화면 확인
source Ros2_ws/install/setup.bash
ros2 run result_subscriber result_subscriber_node
```

ROS2 가 없으면 발행만 꺼지고 GUI 는 그대로 동작합니다.

## 📦 패키지 버전 (개발 당시)

PyQt5 5.15.9 · numpy 1.26.4 · scipy 1.15.3 · Pillow 11.2.1 · opencv-python 4.11.0.86 · torch 2.7.1 · mediapipe 0.10.21 · rclpy 3.3.16 (Ubuntu 22.04)
