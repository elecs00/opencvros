import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import numpy as np
from sensor_msgs.msg import Image
try:
    from cv_bridge import CvBridge
    CVBRIDGE_AVAILABLE = True
except ImportError:
    CVBRIDGE_AVAILABLE = False

def safe_json(data):
    if type(data).__name__ == "NormalizedLandmark":
        return {
            "x": float(data.x),
            "y": float(data.y),
            "z": float(data.z),
            "visibility": float(getattr(data, "visibility", 0.0)),
            "presence": float(getattr(data, "presence", 0.0))
        }
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
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

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'gui_image', 10)
        if CVBRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None

    def publish_image(self, cv_img):
        if not self.bridge:
            self.get_logger().error('cv_bridge가 설치되어 있지 않습니다.')
            return
        msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('Published GUI image')

# 기존 socket_sender.send_result_via_socket(result_dict, host, port) 대신 아래처럼 사용
# 예시:
#   rclpy.init()
#   node = ResultPublisher()
#   node.send_result(result_to_send)
#   rclpy.shutdown()

# 아래는 테스트용 main (실제 사용시 위 함수만 import해서 사용)
def main(args=None):
    rclpy.init(args=args)
    node = ResultPublisher()
    try:
        import time
        # 실제 사용 예시: result_to_send를 반복적으로 publish
        for _ in range(5):
            # 예시 result_dict (실제 분석 결과로 대체)
            result_to_send = {
                'yolo': [{'bbox': [0,0,100,100], 'conf': 0.9, 'class_id': 0, 'class_name': 'person'}],
                'dlib': {'is_drowsy_ear': True, 'is_yawning': False},
                'mediapipe': {'is_drowsy': False},
                'openvino': {},
                'status': 'NORMAL'
            }
            node.send_result(result_to_send)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
