//
//  main.m
//  mlcompute-test
//
//  Created by Wang Zhang on 7/28/20.
//

#include <vector>
#include <iostream>
#import <Foundation/Foundation.h>
#import <MLCompute/MLCompute.h>

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

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        std::vector<int> shape_vec{6,1};
        MLCTensor *tensor1 = CreateMLCTensor(shape_vec, MLCDataTypeFloat32);
        MLCTensor *tensor2 = CreateMLCTensor(shape_vec, MLCDataTypeFloat32);
        MLCTensor *tensor3 = CreateMLCTensor(shape_vec, MLCDataTypeFloat32);
        
        std::vector<Float32> v1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<Float32> v2{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
        std::vector<Float32> v3{-0.1, -0.1, -0.1, -0.1, -0.1, -0.1};
        
        MLCTensorData *data1 = CreateMLCTensorData<Float32>(v1);
        MLCTensorData *data2 = CreateMLCTensorData<Float32>(v2);
        MLCTensorData *data3 = CreateMLCTensorData<Float32>(v3);
                        
        MLCGraph *g = [MLCGraph graph];
        
        MLCArithmeticLayer *arithLayer1 = [MLCArithmeticLayer layerWithOperation:MLCArithmeticOperationAdd];
        MLCArithmeticLayer *arithLayer2 = [MLCArithmeticLayer layerWithOperation:MLCArithmeticOperationAdd];
        
        MLCTensor *im1 = [g nodeWithLayer:arithLayer1 sources:@[tensor1, tensor2]];
        MLCTensor *result_tensor = [g nodeWithLayer:arithLayer2 sources:@[im1, tensor3]];
        
        MLCInferenceGraph *i = [MLCInferenceGraph graphWithGraphObjects:@[g]];
        [i addInputs:@{@"data1":tensor1, @"data2":tensor2, @"data3":tensor3}];
        
        MLCDevice *device = [MLCDevice deviceWithType:MLCDeviceTypeGPU];
        bool compiling_result = [i compileWithOptions:MLCGraphCompilationOptionsNone device:device];
        if (compiling_result) {
            std::cout << "compiling succeeded" << std::endl;
        } else {
            std::cout << "compiling failed" << std::endl;
            return 1;
        }
        
        BOOL *execution_finished = (BOOL*)malloc(sizeof(BOOL));
        *execution_finished = false;
        
        bool execute_result = [i executeWithInputsData:@{@"data1":data1, @"data2":data2, @"data3":data3} batchSize:0 options:MLCExecutionOptionsSynchronous completionHandler:^(MLCTensor * _Nullable __autoreleasing resultTensor, NSError * _Nullable error, NSTimeInterval executionTime) {
            if (error == NULL) {
                NSLog(@"executing succeeded");
            } else {
                NSLog(@"error when executing: %@", error);
            }
            
            *execution_finished = true;
            
        }];
                
        if (execute_result) {
            std::cout << "execution scheduling succeeded" << std::endl;
        } else {
            std::cout << "execution scheduling failed" << std::endl;
            return 1;
        }
        
        while (!(*execution_finished)) {
            [NSThread sleepForTimeInterval:0.1f];
        }
        PrintTensor<Float32>(result_tensor);
        
        
    }
    return 0;
}

