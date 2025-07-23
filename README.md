# Mediapipe Only Project

This project utilizes Mediapipe for analysis and integrates with ROS2 for data publishing and subscribing.

## Files:
- `config_manager.py`: Manages configuration settings.
- `config.json`: Configuration file.
- `detector_utils.py`: Utility functions for detectors.
- `gui_app.py`: Main GUI application.
- `gui_state.json`: Stores GUI state.
- `mediapipe_analyzer.py`: Performs Mediapipe analysis.
- `visualizer.py`: Visualizes results.

## ROS2 Integration:
- `Ros2_ws/src/result_publisher/result_publisher/publisher_node.py`: ROS2 publisher node for results.
- `Ros2_ws/src/result_subscriber/src/result_subscriber_node.cpp`: ROS2 subscriber node for images.

## System Requirements:
- **Operating System**: Linux (Ubuntu 22.04 LTS recommended)
- **Hardware**: A dedicated GPU is recommended for optimal performance with MediaPipe and PyTorch.

## Installation:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/elecs00/opencvros.git
   cd opencvros
   ```
2. **Create a Python virtual environment (recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: A `requirements.txt` file is not provided in the repository. You will need to create one manually based on the Python Package Versions listed below.)

4. **Install ROS2 (Humble recommended)**:
   Follow the official ROS2 Humble installation guide for your specific Linux distribution:
   [https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

5. **Build ROS2 workspace**:
   ```bash
   cd Ros2_ws
   rosdep install -i --from-path src --rosdistro humble -y
   colcon build
   source install/setup.bash
   ```

## Python Package Versions:
- `PyQt5`: 5.15.9
- `numpy`: 1.26.4
- `scipy`: 1.15.3
- `Pillow`: 11.2.1
- `opencv-python`: 4.11.0.86
- `torch`: 2.7.1
- `rclpy`: 3.3.16
- `mediapipe`: 0.10.21