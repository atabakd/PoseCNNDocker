#!/bin/sh
apt update
apt install x11-xserver-utils -y
docker build --rm -t posecnn:v1 .
xhost +
sudo nvidia-docker run -it  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --rm posecnn:v1
