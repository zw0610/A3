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
NSArray<coreType *> * to_nsarray(std::vector<pyType*> sources) {
    NSMutableArray * ma = [[NSMutableArray alloc] init];
    for (size_t i = 0; i<sources.size(); i++) {
        [ma addObject:sources[i]->core];
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



struct tensor {
    MLCTensor * core;

    tensor(MLCTensor * xcore) {
        core = xcore;
    }

    tensor(const tensor& x) {
        core = x.core;
    }

    tensor(std::vector<int> shape, const std::string &dtype) {
        if (dtype != "float32" && dtype != "Float32") {
            std::cout << "warning: only float32 is supported, converting to float32" << std::endl;
        }
        core = CreateMLCTensor(shape, MLCDataTypeFloat32);
    }

    void print(void) {
        PrintTensor<Float32>(core);
    }
};

struct tensor_data {
    MLCTensorData * core;

    tensor_data(py::array_t<Float32> data) {
        size_t len = data.size();
        auto data_info = data.request();

        Float32* data_array = (Float32*)data_info.ptr;

        core = CreateMLCTensorData<Float32>(data_array, len);
    }

    void print(void) {
        PrintTensor<Float32>(core);
    }
};

struct device {
    MLCDevice *core;

    device(const std::string & device_type) {
        if (device_type == "cpu" || device_type == "CPU") {
            core = [MLCDevice deviceWithType:MLCDeviceTypeCPU];
        }
        if (device_type == "gpu" || device_type == "GPU") {
            core = [MLCDevice deviceWithType:MLCDeviceTypeGPU];
        }
    }
};

class layer {
public:
    virtual MLCTensor * _add_self_to_graph(MLCGraph * g, NSArray<MLCTensor *> * sources) = 0;
};

// class py_layer : public layer {
// public:
//     using layer::layer;

//     MLCTensor * _add_self_to_graph(MLCGraph * g, NSArray<MLCTensor *> * sources) override {
//         PYBIND11_OVERLOAD_PURE(
//             MLCTensor *,
//             layer,
//             _add_self_to_graph,
//             MLCGraph *, NSArray<MLCTensor *> *
//         );
//     }
// }

class arithmetic_layer: public layer {
public:
    MLCArithmeticLayer * core;

    arithmetic_layer(const std::string & operation) {
        if (operation == "add") {
            core = [MLCArithmeticLayer layerWithOperation:MLCArithmeticOperationAdd];
        }
    }

    MLCTensor * _add_self_to_graph(MLCGraph * g, NSArray<MLCTensor *> * sources) override {
        return [g nodeWithLayer:core sources:sources];
    }
};

struct graph {
    MLCGraph * core;

    graph() {
        core = [MLCGraph graph];
    }

    tensor add_layer(layer* l, std::vector<tensor*> py_sources) {
        NSArray<MLCTensor *> * tensor_array = to_nsarray<tensor, MLCTensor>(py_sources);
        MLCTensor * out = l->_add_self_to_graph(core, tensor_array);
        return tensor(out);
    }

};

struct inference_graph {
    MLCInferenceGraph * core;

    inference_graph(std::vector<graph*> py_graphs) {

        NSArray<MLCGraph *> * graph_array = to_nsarray<graph, MLCGraph>(py_graphs);
        core = [MLCInferenceGraph graphWithGraphObjects:graph_array];
    }

    void add_inputs(py::dict inputs_dict) {
        NSMutableDictionary *inputs = [[NSMutableDictionary alloc] init];
        for (auto item : inputs_dict) {
            auto key = std::string(py::str(item.first));
            tensor * val = item.second.cast<tensor *>();
            [inputs setObject:val->core forKey:[NSString stringWithUTF8String:key.c_str()]];
        }
        [core addInputs:[inputs copy]];
    }

    bool compile(const device* device) {
        return [core compileWithOptions:MLCGraphCompilationOptionsNone device:device->core];
    }

    bool execute(py::dict inputs_dict, int batch_size) {
        NSMutableDictionary *inputs = [[NSMutableDictionary alloc] init];
        for (auto item : inputs_dict) {
            auto key = std::string(py::str(item.first));
            tensor_data * val = item.second.cast<tensor_data *>();
            [inputs setObject:val->core forKey:[NSString stringWithUTF8String:key.c_str()]];
        }
        return [core executeWithInputsData:[inputs copy] batchSize:batch_size options:MLCExecutionOptionsSynchronous completionHandler:nil];
    }
};


PYBIND11_MODULE(mlcompute, mlc) {
    mlc.doc() = "pybind11 objective-c++ mixing test";

    py::class_<tensor>(mlc, "tensor")
        .def(py::init<const tensor &>())
        .def(py::init<std::vector<int>, const std::string &>())
        .def("print", &tensor::print);

    py::class_<tensor_data>(mlc, "tensor_data")
        .def(py::init<py::array_t<Float32>>())
        .def("print", &tensor_data::print);

    py::class_<device>(mlc, "device")
        .def(py::init<const std::string &>());

    py::class_<layer>(mlc, "layer");

    py::class_<arithmetic_layer, layer>(mlc, "arithmetic_layer")
        .def(py::init<const std::string &>());

    py::class_<graph>(mlc, "graph")
        .def(py::init<>())
        .def("add_layer", &graph::add_layer);

    py::class_<inference_graph>(mlc, "inference_graph")
        .def(py::init< std::vector<graph*> >())
        .def("add_inputs", &inference_graph::add_inputs)
        .def("compile", &inference_graph::compile)
        .def("execute", &inference_graph::execute);
        
}
