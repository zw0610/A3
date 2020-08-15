#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <vector>
#include <string>

#import <Foundation/Foundation.h>
#import <MLCompute/MLCompute.h>

namespace py = pybind11;

auto to_cstr = [](NSString * nss){return [nss UTF8String];};

template<typename pyType, typename coreType>
NSArray<coreType *> * to_nsarray(std::list<pyType&> sources) {
    NSMutableArray * ma = [[NSMutableArray alloc] init];
    for (size_t i = 0; i<sources.size(); i++) {
        [ma addObject:sources[i].core];
    }
    return [ma copy];
}

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
MLCTensorData* CreateMLCTensorData(T* data, size_t len) {
    return [MLCTensorData dataWithImmutableBytesNoCopy:data length:len*sizeof(T)];
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

    mlc_tensor(MLCTensor * xcore) {
        core = xcore;
    }

    mlc_tensor(const mlc_tensor& x) {
        core = x.core;
    }

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

struct mlc_tensor_data {
    MLCTensorData * core;

    mlc_tensor_data(py::array_t<Float32> data) {
        size_t len = data.size();
        auto data_info = data.request();

        Float32* data_array = (Float32*)data_info.ptr;

        core = CreateMLCTensorData<Float32>(data_array, len);
    }

    void print(void) {
        PrintTensor<Float32>(core);
    }
};

struct mlc_device {
    MLCDevice *core;

    mlc_device(const std::string & device_type) {
        if (device_type == "cpu" || device_type == "CPU") {
            core = [MLCDevice deviceWithType:MLCDeviceTypeCPU];
        }
        if (device_type == "gpu" || device_type == "GPU") {
            core = [MLCDevice deviceWithType:MLCDeviceTypeGPU];
        }
    }
};

// class mlc_layer {
// public:
//     MLCLayer * base_core;
//     virtual MLCLayer * core_ptr(void);
//     MLCLayer * base_core_ptr(void) {return base_core;}
// };

// class mlc_arithmetic_layer {
// public:
//     MLCArithmeticLayer * core;

//     mlc_arithmetic_layer(const std::string & operation) {
//         if (operation == "add") {
//             core = [MLCArithmeticLayer layerWithOperation:MLCArithmeticOperationAdd];
//             base_core = (MLCLayer *)core;
//         }
//     }

//     MLCLayer * core_ptr(void) {
//         return core;
//     }
// };

// struct mlc_graph {
//     MLCGraph * core;

//     mlc_graph() {
//         core = [MLCGraph graph];
//     }

//     mlc_tensor add_layer(mlc_layer& layer, std::list<mlc_tensor&> py_sources) {
//         NSArray<MLCTensor *> * tensor_array = to_nsarray<mlc_tensor, MLCTensor>(py_sources);
//         MLCTensor * out = [core nodeWithLayer:layer.core_ptr(), sources: tensor_array];
//         return mlc_tensor(out);
//     }

// };

// struct mlc_interrence_graph {
//     MLCInferenceGraph * core;

//     mlc_interrence_graph(std::list<mlc_graph&> py_graphs) {

//         NSArray<MLCGraph *> * graph_array = to_nsarray<mlc_graph, MLCGraph>(py_graphs);
//         core = [MLCInferenceGraph graphWithGraphObjects:graph_array];
//     }

//     void add_inputs(py::dict inputs_dict) {
//         NSMutableDictionary *inputs = [[NSMutableDictionary alloc] init];
//         for (auto item : inputs_dict) {
//             [inputs setObject:item.second.core forKey:[NSString stringWithUTF8String:item.first.c_str()]]
//         }
//         [core addInputs:[inputs copy]];
//     }

//     bool compile(const mlc_device& device) {
//         return [i compileWithOptions:MLCGraphCompilationOptionsNone device:device.core];
//     }

//     bool execute(py::dict input_data, int batch_size) {
//         NSMutableDictionary *inputs = [[NSMutableDictionary alloc] init];
//         for (auto item : inputs_dict) {
//             [inputs setObject:item.second.core forKey:[NSString stringWithUTF8String:item.first.c_str()]]
//         }
//         return [core executeWithInputsData:[inputs copy] batchSize:base_core_ptr options:MLCExecutionOptionsSynchronous completionHandler:nil];
//     }
// };


PYBIND11_MODULE(mlcompute, mlc) {
    mlc.doc() = "pybind11 objective-c++ mixing test";

    py::class_<mlc_tensor>(mlc, "mlc_tensor")
        .def(py::init<const mlc_tensor &>())
        .def(py::init<std::vector<int>, const std::string &>())
        .def("print", &mlc_tensor::print);

    py::class_<mlc_tensor_data>(mlc, "mlc_tensor_data")
        .def(py::init<py::array_t<Float32>>())
        .def("print", &mlc_tensor_data::print);

    py::class_<mlc_device>(mlc, "mlc_device")
        .def(py::init<const std::string &>());
}
