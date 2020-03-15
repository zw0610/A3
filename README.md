# A3

I'm trying to enable DL training on macOS device with accelerators, such as AMD and Intel GPUs.

## Strategy

Metal -> Objective-C++ -> pybind11 -> Numpy

Metal -> Objective-C++ -> Aten -> PyTorch 

## Steps

### POC

convert *MetalComputeBasic* example from Apple into a Python binding one

### PyTorch with Metal

1. ATen storage related APIs
2. Metal version for arithemtic ops
3. Python APIs for Metal
