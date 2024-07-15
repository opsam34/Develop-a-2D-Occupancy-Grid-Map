import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class WallFollowingRobot(Node):
    def __init__(self):
        super().__init__('wall_following_robot')

        self.bridge = CvBridge()

        self.subscriber_top_right = self.create_subscription(
            Image,
            '/overhead_camera/overhead_camera1/image_raw',
            self.callback_top_right,
            10)
        
        self.subscriber_bottom_left = self.create_subscription(
            Image,
            '/overhead_camera/overhead_camera2/image_raw',
            self.callback_bottom_left,
            10)
        
        self.subscriber_bottom_right = self.create_subscription(
            Image,
            '/overhead_camera/overhead_camera3/image_raw',
            self.callback_bottom_right,
            10)
        
        self.subscriber_top_left = self.create_subscription(
            Image,
            '/overhead_camera/overhead_camera4/image_raw',
            self.callback_top_left,
            10)

        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.image_top_right = None
        self.image_bottom_left = None
        self.image_bottom_right = None
        self.image_top_left = None

        self.timer = self.create_timer(1.0, self.control_robot)

        self.map_image = None

    def callback_top_right(self, msg):
        self.image_top_right = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def callback_bottom_left(self, msg):
        self.image_bottom_left = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def callback_bottom_right(self, msg):
        self.image_bottom_right = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def callback_top_left(self, msg):
        self.image_top_left = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def control_robot(self):
        if self.image_top_right is not None and self.image_bottom_left is not None and self.image_bottom_right is not None and self.image_top_left is not None:
            self.process_images()
            self.follow_wall()

    def process_images(self):
        h1, w1, _ = self.image_top_right.shape if self.image_top_right is not None else (0, 0, 3)
        h2, w2, _ = self.image_bottom_left.shape if self.image_bottom_left is not None else (0, 0, 3)
        h3, w3, _ = self.image_bottom_right.shape if self.image_bottom_right is not None else (0, 0, 3)
        h4, w4, _ = self.image_top_left.shape if self.image_top_left is not None else (0, 0, 3)

        total_height = max(h1 + h2, h3 + h4)
        total_width = max(w1 + w4, w2 + w3)

        combined_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        if self.image_top_right is not None:
            combined_image[0:h1, total_width-w1:] = self.image_top_right
        if self.image_bottom_left is not None:
            combined_image[h1:h1+h2, 0:w2] = self.image_bottom_left
        if self.image_bottom_right is not None:
            combined_image[h1+h2:h1+h2+h3, total_width-w3:] = self.image_bottom_right
        if self.image_top_left is not None:
            combined_image[0:h4, 0:w4] = self.image_top_left
        grayscale_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
	_, binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)
        if self.map_image is None or self.map_image.shape != grayscale_image.shape:
            self.map_image = np.ones_like(grayscale_image) * 255

        self.map_image[binary_image == 0] = 0
        if int(time.time()) % 10 == 0:
            cv2.imwrite('map.pgm', self.map_image)
            self.get_logger().info('Map saved as map.pgm')
    def follow_wall(self):
        cmd_vel = Twist()

        if self.image_top_left is not None:
            left_wall_distance = np.mean(self.image_top_left[:, -1])
            if left_wall_distance < 100:
                cmd_vel.angular.z = 0.3
            else:
                cmd_vel.angular.z = 0.0

        cmd_vel.linear.x = 0.2

        self.publisher_cmd_vel.publish(cmd_vel)

        self.get_logger().info(f'Published velocity command: Linear x: {cmd_vel.linear.x}, Angular z: {cmd_vel.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = WallFollowingRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


