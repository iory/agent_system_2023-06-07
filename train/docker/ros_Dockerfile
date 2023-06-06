# Use the official ubuntu:18.04 image as the parent image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
SHELL ["bash", "-c"]

ENV WORKSPACE /catkin_ws

ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update && apt install -y curl software-properties-common

RUN apt update && apt install -y \
    pkg-config \
    ninja-build \
    doxygen \
    clang \
    gcc-multilib \
    g++-multilib \
    python3 \
    git-lfs \
    nasm \
    cmake \
    libgl1-mesa-dev \
    libsoundio-dev \
    libvulkan-dev \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libusb-1.0-0-dev \
    libssl-dev \
    libudev-dev \
    mesa-common-dev \
    uuid-dev \
    libopencv-dev \
    expect \
    python-pip \
    wget

######### INSTALL ROS START ############
# install packages
RUN apt-get update && apt-get install -q -y \
  dirmngr \
  gnupg2 \
  lsb-release

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  python-rosdep \
  python-rosinstall \
  python-vcstools \
  vim \
  tmux

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
  && rosdep update

# install ros packages
ENV ROS_DISTRO melodic
RUN apt-get update && apt-get install -y \
  ros-melodic-ros-core=1.4.1-0*
######### INSTALL ROS END ############

RUN apt-get update && apt-get install --no-install-recommends -y \
    python-catkin-tools

RUN rosdep update \
    && apt update

RUN mkdir ~/ros/jsk_demo_ws/src -p \
    && cd ~/ros/jsk_demo_ws \
    && wstool init src \
    && wstool merge -t src https://raw.githubusercontent.com/jsk-ros-pkg/jsk_demos/master/jsk_2013_04_pr2_610/jsk_2013_04_pr2_610.rosinstall \
    && wstool merge -t src https://raw.githubusercontent.com/HiroIshida/jsk_pr2eus/update-tutorial-readme-melodic/pr2eus_tutorials/pr2eus_tutorials.rosinstall \
    && wstool update -t src \
    && rosdep install --from-paths src --ignore-src -r -n -y || true

RUN mv /bin/sh /bin/sh_tmp && ln -s /bin/bash /bin/sh
RUN cd ~/ros/jsk_demo_ws && source /opt/ros/${ROS_DISTRO}/setup.bash; catkin build -DCMAKE_BUILD_TYPE=Release jsk_2013_04_pr2_610 detect_cans_in_fridge_201202 pr2eus_tutorials
RUN rm /bin/sh && mv /bin/sh_tmp /bin/sh

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
  ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
  ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

CMD ["bash"]
