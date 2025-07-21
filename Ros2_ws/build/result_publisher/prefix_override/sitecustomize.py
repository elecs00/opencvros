import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hkit/Downloads/test/mediapipe_only/Ros2_ws/install/result_publisher'
