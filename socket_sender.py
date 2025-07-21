import socket
import json
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String

def safe_json(data):
    # mediapipe NormalizedLandmark 변환 (정확한 타입명으로 체크)
    if type(data).__name__ == "NormalizedLandmark":
        return {
            "x": float(data.x),
            "y": float(data.y),
            "z": float(data.z),
            "visibility": float(getattr(data, "visibility", 0.0)),
            "presence": float(getattr(data, "presence", 0.0))
        }
    # numpy array/number 처리
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    # dict, list, tuple 처리
    elif isinstance(data, dict):
        return {k: safe_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [safe_json(v) for v in data]
    elif isinstance(data, tuple):
        return [safe_json(v) for v in data]
    else:
        return data

class ResultPublisher(Node):
    def __init__(self):
        super().__init__('result_publisher')
        self.publisher_ = self.create_publisher(String, 'result_topic', 10)

    def send_result(self, result_dict):
        safe_data = safe_json(result_dict)
        json_data = json.dumps(safe_data)
        msg = String()
        msg.data = json_data
        self.publisher_.publish(msg)
        self.get_logger().info('Published result')

# 사용 예시 (기존 socket_sender.send_result_via_socket 대신)
# publisher = ResultPublisher()
# publisher.send_result(result_to_send)