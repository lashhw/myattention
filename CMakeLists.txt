cmake_minimum_required(VERSION 3.25)
project(tflm)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_USE_CTIME)

add_executable(tflm main.cpp my_attention_model.cpp)
target_include_directories(tflm PRIVATE tflite-micro)
target_include_directories(tflm PRIVATE tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include)
target_include_directories(tflm PRIVATE tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp)
target_link_directories(tflm PRIVATE tflite-micro/gen/linux_x86_64_debug/lib)
target_link_libraries(tflm tensorflow-microlite)