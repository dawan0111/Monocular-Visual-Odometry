# Simple Monocular Visual Odometry

[![gif.gif](https://i.postimg.cc/CK5PV13m/2024-07-13-23-26-08.png)](https://youtu.be/MWPI8KHWMYU)

[[🔗 Video Link]](https://youtu.be/MWPI8KHWMYU)

## Overview
This project is a simplified implementation of Monocular Visual Odometry that operates within the ROS2 environment. It uses OpenCV, Eigen3, and Sophus libraries to perform visual odometry using only essential functionalities.

## Features
- **Monocular Visual Odometry**: Utilizes a single-camera setup to estimate the motion of the camera by analyzing the changes in the images.
- **ROS2 Integration**: Fully compatible with ROS2, allowing for easy deployment and integration with other ROS2 components.
- **Library Usage**: Makes use of OpenCV for image processing, Eigen3 for matrix and vector operations, and Sophus for Lie groups and manifold operations related to 3D motions.

## Prerequisites
Before you can run this project, you need to have ROS2 installed on your system. Additionally, ensure that OpenCV, Eigen3, and Sophus libraries are properly installed and configured in your ROS2 environment.

## Installation
To get started with the Simple Monocular Visual Odometry, clone this repository into your ROS2 workspace:

```bash
git clone https://github.com/dawan0111/Monocular-Visual-Odometry
cd Monocular-Visual-Odometry
```

## Usage
To launch the Monocular Visual Odometry node, use the following command:

```bash
ros2 launch mono_visual_odometry mono_visual_odometry_node
```

This command will start the visual odometry node and begin processing the image data from your camera.

## Contributing
Contributions to the project are welcome. Please fork the repository, make your changes, and submit a pull request for review.

## License
This project is released under the [MIT License](LICENSE).