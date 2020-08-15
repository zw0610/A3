#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <string>
#include <assert.h>
#include <stdlib.h>

#include <algorithm>
#include <iterator>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <ComputingMetal/ComputingMetal.h>

namespace py = pybind11;

auto to_cstr = [](NSString * nss){return [nss UTF8String];};

MLCTensor* CreateMLCTensor(std::vector<int> &shape_vec, MLCDataType dtype) {
    NSMutableArray *shape_nsma = [[NSMutableArray alloc] init];
    for (int i = 0; i < shape_vec.size(); i++) {
        [shape_nsma addObject:[NSNumber numberWithInt:shape_vec[i]]];
    }
    NSArray *shape_nsa = [shape_nsma copy];
    MLCTensorDescriptor *md = [MLCTensorDescriptor descriptorWithShape:shape_nsa dataType:dtype];
    return [MLCTensor tensorWithDescriptor:md];
}

template<typename T>
MLCTensorData* CreateMLCTensorData(std::vector<T> &data) {
    return [MLCTensorData dataWithImmutableBytesNoCopy:data.data() length:data.size()*sizeof(T)];
}

template<typename T>
void PrintTensor(MLCTensorData * td) {
    const T* ptr = (const T*)td.bytes;
    for (size_t i = 0; i<(td.length/sizeof(T)); i++) {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void PrintTensor(MLCTensor *t) {
    const int bytes_count = (const int)t.descriptor.tensorAllocationSizeInBytes;
    const int element_count = bytes_count / sizeof(T);
    std::vector<T> temp(element_count);
    [t copyDataFromDeviceMemoryToBytes:temp.data() length:bytes_count synchronizeWithDevice:true];
    std::cout << to_cstr(t.className) << " " << to_cstr(t.label) << " : ";
    for (auto x : temp) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

struct mlc_tensor {
    MLCTensor *

    objcpp_test() {
        buff_a = nullptr;
        buff_b = nullptr;
        buff_c = nullptr;
        len_a = 0;
        len_b = 0;
        len_c = 0;

        device = MTLCreateSystemDefaultDevice();

        adder = [[MetalAdder alloc] initWithDevice:device];
    }

    void print_device() {
        NSLog(@"Using device: %@\n", device.name);
        return;
    }

    void run_metal(py::array_t<float> a, py::array_t<float> b, py::array_t<float> r) {
        unsigned int len = a.size();

        assert(a.size() == b.size());
        assert(a.size() == r.size());

        auto a_info = a.request();
        auto b_info = b.request();
        auto r_info = r.request();

        float* array_a = (float*)a_info.ptr;
        float* array_b = (float*)b_info.ptr;
        float* array_r = (float*)r_info.ptr;

        [adder sendComputeCommand:len array_a:array_a array_b:array_b array_r:array_r];
        
        NSLog(@"Execution finished.");
    }


    
};

PYBIND11_MODULE(mlcompute-cpp-bridge, m) {
    m.doc() = "pybind11 objective-c++ mixing test";

    py::class_<objcpp_test>(m, "objcpp_test")
        .def(py::init<>())
        .def("print_device", &objcpp_test::print_device)
        .def("run_metal", &objcpp_test::run_metal);
}
