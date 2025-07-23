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

## Python Package Versions:
- `PyQt5`: 5.15.9
- `numpy`: 1.26.4
- `scipy`: 1.15.3
- `Pillow`: 11.2.1
- `opencv-python`: 4.11.0.86
- `torch`: 2.7.1
- `rclpy`: 3.3.16
- `mediapipe`: 0.10.21