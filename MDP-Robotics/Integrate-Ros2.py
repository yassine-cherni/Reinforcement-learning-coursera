# mdp_node.py
import rclpy
from mdp_framework import MDPFramework

def main():
    rclpy.init()
    node = rclpy.create_node('mdp_node')

    mdp_framework = MDPFramework()

    # Add ROS2 communication logic here if needed

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
  
