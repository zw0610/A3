#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <stdlib.h>

#include <algorithm>
#include <iterator>

#import <Foundation/Foundation.h>
#import <MLCompute/MLCompute.h>

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
    MLCTensor * core;

    mlc_tensor(std::vector<int> shape, const std::string &dtype) {
        if (dtype != "float32" && dtype != "Float32") {
            std::cout << "warning: only float32 is supported, converting to float32" << std::endl;
        }

        core = CreateMLCTensor(shape, MLCDataTypeFloat32);
    }

    void print(void) {
        PrintTensor<Float32>(core);
    }
};

PYBIND11_MODULE(mlcompute, mlc) {
    mlc.doc() = "pybind11 objective-c++ mixing test";

    py::class_<mlc_tensor>(mlc, "mlc_tensor")
        .def(py::init<std::vector<int>, const std::string &>())
        .def("print", &mlc_tensor::print);
}
