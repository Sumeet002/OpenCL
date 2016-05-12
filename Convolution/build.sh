#!/bin/bash

echo "OpenCL Convolution example"

echo "Building the OpenCL convolution example "
g++ -O2  convolution.cpp `pkg-config --libs opencv` -lOpenCL -o convolution

echo "Building the OpenCV convolution example "
g++ -O2  convolution_verify.cpp `pkg-config --libs opencv` -o convolution_verify

echo "Finished"
