xcrun -sdk macosx metal -c ./arithmetic.metal
xcrun -sdk macosx metallib arithmetic.air
mv ./default.metallib ./arithmetic.metallib
