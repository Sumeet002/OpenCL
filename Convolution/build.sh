#!/bin/bash

echo "OpenCL Convolution example"

echo "Building the OpenCL convolution example "
g++ -O2  convolution.cpp `pkg-config --libs opencv` -lOpenCL -o boneCV

echo "Finished"
