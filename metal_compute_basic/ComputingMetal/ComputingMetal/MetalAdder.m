/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import "MetalAdder.h"

// The number of floats in each array, and the size of the arrays in bytes.
//const unsigned int arrayLength = 1 << 24;
//const unsigned int bufferSize = arrayLength * sizeof(float);

@implementation MetalAdder
{
    id<MTLDevice> _mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mAddFunctionPSO;

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;

//    // Buffers to hold data.
//    id<MTLBuffer> _mBufferA;
//    id<MTLBuffer> _mBufferB;
//    id<MTLBuffer> _mBufferResult;

}

- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super init];
    if (self)
    {
        _mDevice = device;

        NSError* error = nil;
        NSError* error_newlib = nil;
        
        NSString* filepath = @"/Users/lucas/Projects/arithmetic/arithmetic/arithmetic.metallib";

        // Load the shader files with a .metal file extension in the project

        id<MTLLibrary> defaultLibrary = [_mDevice newLibraryWithFile:filepath error:&error_newlib];
        if (defaultLibrary == nil)
        {
            NSLog(@"Failed to find the library, error %@.", error_newlib);
            return nil;
        }

        id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"add_arrays"];
        if (addFunction == nil)
        {
            NSLog(@"Failed to find the adder function.");
            return nil;
        }

        // Create a compute pipeline state object.
        _mAddFunctionPSO = [_mDevice newComputePipelineStateWithFunction: addFunction error:&error];
        if (_mAddFunctionPSO == nil)
        {
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode)
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return nil;
        }

        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }
    }

    return self;
}

//- (void) sendComputeCommand
//{
//    // Create a command buffer to hold commands.
//    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
//    assert(commandBuffer != nil);
//
//    // Start a compute pass.
//    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
//    assert(computeEncoder != nil);
//
//    [self encodeAddCommand:computeEncoder];
//
//    // End the compute pass.
//    [computeEncoder endEncoding];
//
//    // Execute the command.
//    [commandBuffer commit];
//
//    // Normally, you want to do other work in your app while the GPU is running,
//    // but in this example, the code simply blocks until the calculation is complete.
//    [commandBuffer waitUntilCompleted];
//
//    [self verifyResults];
//}

- (void) sendComputeCommand:(const unsigned int) len
                    array_a:(const float*) array_a
                    array_b:(const float*) array_b
                    array_r:(float*) array_r
{
    const unsigned int bytes = len * sizeof(float);
    id<MTLBuffer> buffer_a = [_mDevice newBufferWithBytes:array_a length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_b = [_mDevice newBufferWithBytes:array_b length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_r = [_mDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);
    
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);
    
    [self encodeAddCommand:computeEncoder bufferA:buffer_a bufferB:buffer_b bufferR:buffer_r length:len];
    
    [computeEncoder endEncoding];
    
    [commandBuffer commit];
    
    [commandBuffer waitUntilCompleted];
    
    assert(array_r != NULL);
    float * buffer_r_arr = buffer_r.contents;
    for (unsigned long index = 0; index < len; index++)
    {
        array_r[index] = buffer_r_arr[index];
    }
}

- (void)encodeAddCommand:(id<MTLComputeCommandEncoder>)computeEncoder
                 bufferA:(id<MTLBuffer>) buffer_a
                 bufferB:(id<MTLBuffer>) buffer_b
                 bufferR:(id<MTLBuffer>) buffer_r
                  length:(const unsigned int) len
{
    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
    
    [computeEncoder setBuffer:buffer_a offset:0 atIndex:0];
    [computeEncoder setBuffer:buffer_b offset:0 atIndex:1];
    [computeEncoder setBuffer:buffer_r offset:0 atIndex:2];
    
    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    
    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > len) {
        threadGroupSize = len;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

//- (void)encodeAddCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
//
//    // Encode the pipeline state object and its parameters.
//    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
//    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
//    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
//    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];
//
//    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);
//
//    // Calculate a threadgroup size.
//    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
//    if (threadGroupSize > arrayLength)
//    {
//        threadGroupSize = arrayLength;
//    }
//    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
//
//    // Encode the compute command.
//    [computeEncoder dispatchThreads:gridSize
//              threadsPerThreadgroup:threadgroupSize];
//}

@end
