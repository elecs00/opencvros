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
