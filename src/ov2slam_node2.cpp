/**
 *  OV²SLAM mono ROS 2 node (ov2slam2_node2)
 *  - Mono-only input
 *  - Configurable image topic via param: left_topic (default /image)
 *  - Param file via argv[1] OR ROS param: param_file
 *  - SensorData QoS for camera stream
 *  - Robust image encoding → grayscale conversion
 *
 *  This file depends on libov2slam (your core library) which expects a global
 *  std::shared_ptr<rclcpp::Node> named 'nh'. We define and assign it here.
 */

#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>   // cv::cvtColor
#include <opencv2/core.hpp>      // cv::FileStorage

// Optional (only needed if your core publishes TF directly)
// #include <tf2_msgs/msg/tf_message.hpp>

#include "ov2slam.hpp"
#include "slam_params.hpp"

// ---- Global node pointer expected by some core code (e.g., visualizer/TF) ----
std::shared_ptr<rclcpp::Node> nh;

// ---- Mono grabber ------------------------------------------------------------
class MonoGrabber {
public:
  explicit MonoGrabber(SlamManager* slam) : pslam_(slam) {}

  void subImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(m_);
    buf_.push(msg);
  }

  static cv::Mat toGray(const sensor_msgs::msg::Image::SharedPtr& msg) {
    // Try direct MONO8 via cv_bridge
    try {
      auto cvp = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
      return cvp->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN(rclcpp::get_logger("ov2slam_node2"),
                  "cv_bridge MONO8 failed: %s. Trying manual convert...", e.what());
    }

    // Fallback: native encoding → convert to gray
    try {
      auto cvp = cv_bridge::toCvCopy(msg);  // native encoding
      const std::string enc = msg->encoding;
      cv::Mat gray;

      if (enc == sensor_msgs::image_encodings::BGR8) {
        cv::cvtColor(cvp->image, gray, cv::COLOR_BGR2GRAY);
      } else if (enc == sensor_msgs::image_encodings::RGB8) {
        cv::cvtColor(cvp->image, gray, cv::COLOR_RGB2GRAY);
      } else if (enc == sensor_msgs::image_encodings::YUV422_YUY2 ||
                 enc == "yuyv" || enc == "YUYV" || enc == "yuy2" || enc == "YUY2" ||
                 enc == "yuv422_yuy2" || enc == "YUV422_YUY2") {
        cv::cvtColor(cvp->image, gray, cv::COLOR_YUV2GRAY_YUY2);
      } else if (enc == "uyvy" || enc == "UYVY" ||
                 enc == "yuv422_uyvy" || enc == "YUV422_UYVY") {
        // Humble doesn’t define a UYVY constant; match strings and convert.
        cv::cvtColor(cvp->image, gray, cv::COLOR_YUV2GRAY_UYVY);
      } else {
        // Assume already single-channel or unknown but usable
        gray = cvp->image;
        if (gray.channels() != 1) {
          RCLCPP_WARN(rclcpp::get_logger("ov2slam_node2"),
                      "Unknown encoding '%s' with %d channels; using first channel.",
                      enc.c_str(), gray.channels());
          std::vector<cv::Mat> ch; cv::split(gray, ch); gray = ch[0];
        }
      }
      return gray;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("ov2slam_node2"),
                   "cv_bridge native conversion failed: %s", e.what());
      return {};
    }
  }

  void runSyncLoop() {
    RCLCPP_INFO(rclcpp::get_logger("ov2slam_node2"), "Sync loop started.");
    while (!pslam_->bexit_required_) {
      cv::Mat img;
      double stamp = 0.0;
      {
        std::lock_guard<std::mutex> lk(m_);
        if (!buf_.empty()) {
          auto m = buf_.front(); buf_.pop();
          stamp = static_cast<double>(m->header.stamp.sec)
                + static_cast<double>(m->header.stamp.nanosec) * 1e-9;
          img = toGray(m);
        }
      }
      if (!img.empty()) {
        pslam_->addNewMonoImage(stamp, img);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    RCLCPP_INFO(rclcpp::get_logger("ov2slam_node2"), "Sync loop exiting.");
  }

private:
  std::mutex m_;
  std::queue<sensor_msgs::msg::Image::SharedPtr> buf_;
  SlamManager* pslam_;
};

// ---- main --------------------------------------------------------------------
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("ov2slam_node2");

  // make global 'nh' visible to library code (visualizer, TF, etc.)
  nh = node;

  // Params
  node->declare_parameter<std::string>("param_file", "");
  node->declare_parameter<std::string>("left_topic", "/image"); // set to /image_raw if needed

  // Resolve param file: argv[1] (if not a ROS arg) OR param
  std::string param_file;
  if (argc >= 2 && std::string(argv[1]).rfind("--", 0) != 0) {
    param_file = argv[1];
  } else {
    param_file = node->get_parameter("param_file").as_string();
  }
  if (param_file.empty()) {
    std::cerr << "\nUsage:\n"
              << "  ros2 run ov2slam ov2slam_node2 -- </path/to/params.yaml> [--ros-args ...]\n"
              << "  or\n"
              << "  ros2 run ov2slam ov2slam_node2 --ros-args -p param_file:=</path/to/params.yaml>\n";
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Loading parameters file: %s", param_file.c_str());
  cv::FileStorage fs(param_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_ERROR(node->get_logger(), "Failed to open params file.");
    return 1;
  }

  // Init SLAM
  auto params = std::make_shared<SlamParams>(fs);
  auto viz    = std::make_shared<RosVisualizer>(node);
  SlamManager slam(params, viz);

  // Force mono mode regardless of YAML flags (optional safety)
  if (slam.pslamstate_) {
    slam.pslamstate_->mono_   = true;
    slam.pslamstate_->stereo_ = false;
  }

  std::thread slam_thread(&SlamManager::run, &slam);

  // Subscriber (mono)
  MonoGrabber grab(&slam);
  rclcpp::SensorDataQoS qos; // best-effort, small queue, for camera
  const auto left_topic = node->get_parameter("left_topic").as_string();
  auto sub_left = node->create_subscription<sensor_msgs::msg::Image>(
      left_topic, qos, std::bind(&MonoGrabber::subImage, &grab, std::placeholders::_1));

  // Feeding thread
  std::thread sync_thread(&MonoGrabber::runSyncLoop, &grab);

  rclcpp::spin(node);

  // Shutdown
  slam.bexit_required_ = true;
  if (sync_thread.joinable()) sync_thread.join();
  if (slam_thread.joinable()) slam_thread.join();
  rclcpp::shutdown();
  return 0;
}

