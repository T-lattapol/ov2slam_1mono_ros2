#!/usr/bin/env python3
# video_pub.py — video publisher for OV2SLAM monocamera
#
# Usage:
#   python3 video_pub.py --video input.mp4
#   python3 video_pub.py --video input.mp4 --loop 3 --fps 30 --grayscale
#   python3 video_pub.py --video input.mp4 --loop infinite --image_size 640 480
#
# Options:
#   --video <path>        Path to MP4 file (required)
#   --loop <once|infinite|N>  Loop behavior: play once, loop forever, or play N times
#   --fps <float>         Publishing FPS (default = use video FPS or 30)
#   --topic <name>        ROS 2 topic to publish (default: /image_raw)
#   --frame_id <str>      Frame ID in ROS header (default: camera)
#   --grayscale           Convert frames to grayscale (mono8)
#   --image_size W H      Resize frames to width W and height H

import argparse, os, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoPublisher(Node):
    def __init__(self, video_path, loop_kind, loop_num, fps, topic, frame_id, grayscale, image_size):
        super().__init__('video_pub')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, topic, 10)
        self.loop_kind = loop_kind
        self.loop_num = loop_num
        self.frame_id = frame_id
        self.grayscale = grayscale
        self.image_size = image_size
        self.completed_loops = 0
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        if fps > 0:
            self.dt = 1.0 / fps
        else:
            vfps = self.cap.get(cv2.CAP_PROP_FPS)
            self.dt = 1.0 / vfps if vfps and vfps > 0 else 1.0 / 30.0

        self.timer = self.create_timer(self.dt, self.tick)

    def publish_frame(self, frame):
        # resize if requested
        if self.image_size is not None:
            w, h = self.image_size
            frame = cv2.resize(frame, (w, h))

        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        encoding = 'mono8' if self.grayscale else 'bgr8'
        msg = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.frame_id:
            msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def _loop_finished(self):
        self.completed_loops += 1
        if self.loop_kind == 'infinite':
            return False
        if self.loop_kind == 'once':
            return True
        if self.loop_num > 0 and self.completed_loops >= self.loop_num:
            return True
        return False

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            if self._loop_finished():
                self.get_logger().info("Finished all loops, shutting down.")
                rclpy.shutdown()
                return
            # restart video
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().error("Failed to restart video.")
                rclpy.shutdown()
                return
        self.publish_frame(frame)

def main():
    ap = argparse.ArgumentParser(description="Video publisher for OV2SLAM monocamera")
    ap.add_argument('--video', type=str, required=True, help="Path to MP4")
    ap.add_argument('--loop', default='once', help="Loop behavior: once | infinite | <int>")
    ap.add_argument('--fps', type=float, default=0.0, help="Publish FPS (0 = auto from video)")
    ap.add_argument('--topic', type=str, default='/image_raw', help="ROS 2 image topic")
    ap.add_argument('--frame_id', type=str, default='camera', help="Header frame_id")
    ap.add_argument('--grayscale', action='store_true', help="Convert frames to grayscale")
    ap.add_argument('--image_size', type=int, nargs=2, metavar=('W','H'),
                    help="Resize output frames to width W and height H")
    args = ap.parse_args()

    loop_kind = 'once'
    loop_num = 0
    if args.loop.isdigit():
        loop_kind = 'number'
        loop_num = int(args.loop)
    elif args.loop == 'infinite':
        loop_kind = 'infinite'
    elif args.loop == 'once':
        loop_kind = 'once'
    else:
        ap.error("--loop must be once, infinite, or a positive integer")

    if not os.path.isfile(args.video):
        ap.error(f"Video not found: {args.video}")

    rclpy.init()
    node = VideoPublisher(
        video_path=args.video,
        loop_kind=loop_kind,
        loop_num=loop_num,
        fps=args.fps,
        topic=args.topic,
        frame_id=args.frame_id,
        grayscale=args.grayscale,
        image_size=args.image_size
    )
    try:
        rclpy.spin(node)
    finally:
        if hasattr(node, 'cap'):
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
   #

