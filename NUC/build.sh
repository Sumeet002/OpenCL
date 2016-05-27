#!/bin/bash

echo "OpenCL NUC (Non Uniformity Correction)"

echo "Building the OpenCL NUC example "
g++ -O2  NUC.cpp `pkg-config --libs opencv` -lOpenCL -o NUC

echo "Finished"
