#!/bin/bash

docker build -f ./docker/Dockerfile \
       --rm \
       -t hanging_pose_generator \
       .
