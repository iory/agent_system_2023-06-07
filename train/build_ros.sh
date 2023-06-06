#!/bin/bash

docker build \
       --network=host \
       -t agent_system_fg_ros \
       -f docker/ros_Dockerfile .
