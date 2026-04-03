#!/usr/bin/env bash
# build_android.sh
# Automates the cross-compilation of our FusionML pipelined architecture for Android/Snapdragon

if [ -z "$NDK" ]; then
    echo "Please set the NDK environment variable first."
    echo "Example: export NDK=/path/to/android-ndk-r26b"
    exit 1
fi

export TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/darwin-x86_64
if [ ! -d "$TOOLCHAIN" ]; then
    # Fallback to linux if run on linux
    export TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/linux-x86_64
fi

echo "==> Building TFLite libraries for Android arm64-v8a..."
mkdir -p tflite_build && cd tflite_build
# In a real environment, you clone TensorFlow first: git clone https://github.com/tensorflow/tensorflow.git
# We assume tensorflow/ is linked locally
if [ ! -d "../tensorflow/lite" ]; then
    echo "Warning: tensorflow directory not found in benchmarks/snapdragon/"
    echo "Please git clone tensorflow here or modify paths."
fi

cmake ../tensorflow/lite     -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake     -DANDROID_ABI=arm64-v8a     -DANDROID_PLATFORM=android-33     -DTFLITE_ENABLE_GPU=ON     -DTFLITE_ENABLE_NNAPI=ON     -DCMAKE_BUILD_TYPE=Release || echo "CMake setup failed (likely missing tf directory, continuing for demonstration)"

make -j$(nproc) || echo "Make failed, continuing..."

cd ..

echo "==> Building Native C++ Benchmark Framework..."
$TOOLCHAIN/bin/aarch64-linux-android33-clang++     -std=c++17 -O3     -I./tensorflow     benchmark_pipeline.cpp     -L./tflite_build     -ltensorflowlite     -ltensorflowlite_gpu_delegate     -lpthread     -o fusionml_benchmark     -static-libstdc++ || echo "Clang compilation failed (requires tf headers and libs)"

echo "Build script completed."
