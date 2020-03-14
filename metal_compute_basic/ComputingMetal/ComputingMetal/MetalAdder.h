/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface MetalAdder : NSObject
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) sendComputeCommand:(const unsigned int) len
array_a:(const float*) array_a
array_b:(const float*) array_b
array_r:(float*) array_r;
@end

NS_ASSUME_NONNULL_END
