c++ -iframework /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks \
-framework Foundation -framework MLCompute \
-I /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include \
-L /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib \
-O3 -w -shared -std=c++11 \
-undefined dynamic_lookup `python3 -m pybind11 --includes` \
mlcompute-bridge.mm \
-o mlcompute`python3-config --extension-suffix`
