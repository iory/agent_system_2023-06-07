#!/bin/bash

docker build \
               --network=host \
               -t agent_system_melodic \
               -f Dockerfile .
