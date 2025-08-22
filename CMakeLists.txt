cmake_minimum_required(VERSION 3.5)
project(ov2slam)

# -------- Policies --------
if(POLICY CMP0074)
  # find_package honors <PackageName>_ROOT hints
  cmake_policy(SET CMP0074 NEW)
endif()

# -------- Build setup --------
# Use C++20 (camera_visualizer.hpp uses auto in parameter; inline vars if needed)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native -g")

# Options
option(WITH_OPENCV_CONTRIB "Use OpenCV contrib (xfeatures2d, etc.)" ON)
option(WITH_IBOW_LCD        "Use iBoW-LCD for Loop Closing"         OFF)
option(BUILD_MONO_NODE      "Build mono-only node ov2slam_node2"   ON)

# Choose which OpenCV to bind to (default = system/apt to match cv_bridge)
option(USE_SYSTEM_OPENCV "Prefer system OpenCV (apt) over /usr/local" ON)

if(WITH_OPENCV_CONTRIB)
  message(STATUS "[ov2slam] Using OpenCV contrib modules.")
  add_definitions(-DOPENCV_CONTRIB -DOPENCV_ENABLE_NONFREE)
endif()

# -------- Dependencies --------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

# OpenCV (pin to one install)
if(USE_SYSTEM_OPENCV)
  # Ubuntu/ROS Humble OpenCV (matches cv_bridge 4.5.x)
  set(_opencv_hints /usr/lib/x86_64-linux-gnu/cmake/opencv4)
  find_package(OpenCV REQUIRED PATHS ${_opencv_hints} NO_DEFAULT_PATH)
else()
  # Your custom build in /usr/local
  set(_opencv_hints /usr/local/lib/cmake/opencv4 /usr/local/share/opencv4)
  find_package(OpenCV REQUIRED PATHS ${_opencv_hints} NO_DEFAULT_PATH)
endif()
message(STATUS "[ov2slam] OpenCV version: ${OpenCV_VERSION}")
message(STATUS "[ov2slam] OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "[ov2slam] OpenCV libs: ${OpenCV_LIBS}")

# Eigen / Boost / PCL
find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS system filesystem)
# Prefer a concrete hint over PCL_ROOT (which is ignored under OLD policy)
if(NOT PCL_DIR)
  set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl" CACHE PATH "PCL cmake dir")
endif()
find_package(PCL REQUIRED)

# Third-party bundled
find_package(Ceres REQUIRED PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/ceres-solver/install/)
find_package(Sophus REQUIRED PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus/build/)
find_package(opengv QUIET)

# Optional: iBoW-LCD
if(WITH_IBOW_LCD AND EXISTS "${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/build/liblcdetector.so")
  add_definitions(-DIBOW_LCD)
  set(IBOW_LCD_LIB "${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/build/liblcdetector.so")
  message(STATUS "[ov2slam] iBoW-LCD found — Loop Closer enabled.")
else()
  message(STATUS "[ov2slam] iBoW-LCD not found — Loop Closer disabled.")
endif()

# Validate/augment OpenCV contrib header if requested
if(WITH_OPENCV_CONTRIB)
  find_path(OPENCV_XFEATURES2D_HEADER "opencv2/xfeatures2d.hpp"
            PATHS ${OpenCV_INCLUDE_DIRS} /usr/local/include/opencv4)
  if(NOT OPENCV_XFEATURES2D_HEADER)
    message(WARNING
      "[ov2slam] WITH_OPENCV_CONTRIB=ON but 'opencv2/xfeatures2d.hpp' was not found.\n"
      "          Ensure your OpenCV build includes contrib (and NONFREE if needed),\n"
      "          or set WITH_OPENCV_CONTRIB=OFF to build without those features.")
  else()
    # Some setups need the explicit /usr/local include to resolve xfeatures2d.hpp
    include_directories(${OPENCV_XFEATURES2D_HEADER})
  endif()
endif()

# -------- Core library (ov2slam) --------
add_library(${PROJECT_NAME}
  src/ov2slam.cpp
  src/camera_visualizer.cpp
  src/visual_front_end.cpp
  src/frame.cpp
  src/slam_params.cpp
  src/camera_calibration.cpp
  src/feature_extractor.cpp
  src/feature_tracker.cpp
  src/map_manager.cpp
  src/map_point.cpp
  src/mapper.cpp
  src/multi_view_geometry.cpp
  src/ceres_parametrization.cpp
  src/optimizer.cpp
  src/estimator.cpp
  src/loop_closer.cpp
)

# Require C++20 for targets
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ceres_parametrization
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Ceres_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  image_transport
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  Sophus::Sophus
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  Ceres::ceres
)

if(WITH_IBOW_LCD)
  target_link_libraries(${PROJECT_NAME} ${IBOW_LCD_LIB})
endif()
if(opengv_FOUND)
  target_link_libraries(${PROJECT_NAME} opengv)
endif()

# -------- Nodes / executables --------
add_executable(${PROJECT_NAME}_node src/ov2slam_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})
ament_target_dependencies(${PROJECT_NAME}_node rclcpp sensor_msgs cv_bridge)

if(BUILD_MONO_NODE)
  add_executable(ov2slam_node2 src/ov2slam_node2.cpp)
  target_link_libraries(ov2slam_node2 ${PROJECT_NAME})
  ament_target_dependencies(ov2slam_node2 rclcpp sensor_msgs cv_bridge)
endif()

# -------- Install --------
install(TARGETS
  ${PROJECT_NAME}
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)
if(BUILD_MONO_NODE)
  install(TARGETS ov2slam_node2 DESTINATION lib/${PROJECT_NAME})
endif()
install(DIRECTORY include/ DESTINATION include)

# -------- Export --------
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  image_transport
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
)
ament_package()

