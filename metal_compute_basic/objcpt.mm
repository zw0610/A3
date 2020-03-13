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

// class MetalAdder {
//     id<MTLDevice> _mDevice;
//     // id<MTLComputePipelineState> _mAddFunctionPSO;
//     // id<MTLLCommandQueue> _mCommandQueue;
    
//     // id<MTLBuffer> _mBufferA;
//     // id<MTLBuffer> _mBufferB;
//     // id<MTLBuffer> _mBufferResult;
// public:
//     MetalAdder() {
//         NSLog(@"Default constructor called");
//         _mDevice = MTLCreateSystemDefaultDevice();

//         id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
//         if (defaultLibrary == nil) {
//             NSLog(@"Failed to find the default library.");
//         }
//     }

//     MetalAdder(id<MTLDevice> device) {
//         NSLog(@"Non-default constructor called");
//         _mDevice = device;

//         id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
//         if (defaultLibrary == nil) {
//             NSLog(@"Failed to find the default library.");
//         }
//     }

//     void print_device() {
//         NSLog(@"Using device: %@\n", _mDevice.name);
//     }
// };

struct objcpp_test {
    float *buff_a, *buff_b, *buff_c;
    size_t len_a, len_b, len_c;
    id<MTLDevice> device;

    MetalAdder* adder;


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


    void set_x(py::array_t<float> input_data, const std::string buffer_name) {
        
        size_t& len = buffer_name == "a" ? len_a : len_b; 
        len = input_data.size();

        const unsigned int bytes = len * sizeof(float);

        float*& buff = buffer_name == "a" ? buff_a : buff_b;

        if (buff != nullptr) {
            delete[] buff;
        }

        buff = new float[len];

        auto input_data_info = input_data.request();
        memcpy(buff, input_data_info.ptr, bytes);

        for (size_t i = 0; i<len; i++) {
            std::cout << buff[i] << " ";
        }
        std::cout << std::endl;

        return;
    }

    py::array_t<float> add_a_b() {

        assert(len_a == len_b);
        assert(buff_a != nullptr);
        assert(buff_b != nullptr);

        len_c = len_a;

	    const unsigned int bytes = len_c * sizeof(float);

        if (buff_c != nullptr) {
            delete[] buff_c;
        }
        buff_c = new float[len_c];


        for (size_t i = 0; i < len_c; i++) {
            buff_c[i] = buff_a[i] + buff_b[i];
        }

        auto output_data = py::array_t<float>(len_c);
        auto output_data_info = output_data.request();

        memcpy(output_data_info.ptr, buff_c, bytes);

        return output_data;
    }
};

PYBIND11_MODULE(objcpt, m) {
    m.doc() = "pybind11 objective-c++ mixing test";

    py::class_<objcpp_test>(m, "objcpp_test")
        .def(py::init<>())
        .def("print_device", &objcpp_test::print_device)
        // .def("prepare_data", &objcpp_test::prepare_data)
        .def("run_metal", &objcpp_test::run_metal)
        .def("set_x", &objcpp_test::set_x)
        .def("add_a_b", &objcpp_test::add_a_b);
}
