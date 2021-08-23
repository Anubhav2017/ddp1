################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tensorflow_lite/tensorflow/lite/micro/all_ops_resolver.cc \
../tensorflow_lite/tensorflow/lite/micro/debug_log.cc \
../tensorflow_lite/tensorflow/lite/micro/flatbuffer_utils.cc \
../tensorflow_lite/tensorflow/lite/micro/memory_helpers.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_allocator.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_error_reporter.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_graph.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_interpreter.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_profiler.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_string.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_time.cc \
../tensorflow_lite/tensorflow/lite/micro/micro_utils.cc \
../tensorflow_lite/tensorflow/lite/micro/mock_micro_graph.cc \
../tensorflow_lite/tensorflow/lite/micro/recording_micro_allocator.cc \
../tensorflow_lite/tensorflow/lite/micro/recording_simple_memory_allocator.cc \
../tensorflow_lite/tensorflow/lite/micro/simple_memory_allocator.cc \
../tensorflow_lite/tensorflow/lite/micro/system_setup.cc \
../tensorflow_lite/tensorflow/lite/micro/test_helpers.cc 

CC_DEPS += \
./tensorflow_lite/tensorflow/lite/micro/all_ops_resolver.d \
./tensorflow_lite/tensorflow/lite/micro/debug_log.d \
./tensorflow_lite/tensorflow/lite/micro/flatbuffer_utils.d \
./tensorflow_lite/tensorflow/lite/micro/memory_helpers.d \
./tensorflow_lite/tensorflow/lite/micro/micro_allocator.d \
./tensorflow_lite/tensorflow/lite/micro/micro_error_reporter.d \
./tensorflow_lite/tensorflow/lite/micro/micro_graph.d \
./tensorflow_lite/tensorflow/lite/micro/micro_interpreter.d \
./tensorflow_lite/tensorflow/lite/micro/micro_profiler.d \
./tensorflow_lite/tensorflow/lite/micro/micro_string.d \
./tensorflow_lite/tensorflow/lite/micro/micro_time.d \
./tensorflow_lite/tensorflow/lite/micro/micro_utils.d \
./tensorflow_lite/tensorflow/lite/micro/mock_micro_graph.d \
./tensorflow_lite/tensorflow/lite/micro/recording_micro_allocator.d \
./tensorflow_lite/tensorflow/lite/micro/recording_simple_memory_allocator.d \
./tensorflow_lite/tensorflow/lite/micro/simple_memory_allocator.d \
./tensorflow_lite/tensorflow/lite/micro/system_setup.d \
./tensorflow_lite/tensorflow/lite/micro/test_helpers.d 

OBJS += \
./tensorflow_lite/tensorflow/lite/micro/all_ops_resolver.o \
./tensorflow_lite/tensorflow/lite/micro/debug_log.o \
./tensorflow_lite/tensorflow/lite/micro/flatbuffer_utils.o \
./tensorflow_lite/tensorflow/lite/micro/memory_helpers.o \
./tensorflow_lite/tensorflow/lite/micro/micro_allocator.o \
./tensorflow_lite/tensorflow/lite/micro/micro_error_reporter.o \
./tensorflow_lite/tensorflow/lite/micro/micro_graph.o \
./tensorflow_lite/tensorflow/lite/micro/micro_interpreter.o \
./tensorflow_lite/tensorflow/lite/micro/micro_profiler.o \
./tensorflow_lite/tensorflow/lite/micro/micro_string.o \
./tensorflow_lite/tensorflow/lite/micro/micro_time.o \
./tensorflow_lite/tensorflow/lite/micro/micro_utils.o \
./tensorflow_lite/tensorflow/lite/micro/mock_micro_graph.o \
./tensorflow_lite/tensorflow/lite/micro/recording_micro_allocator.o \
./tensorflow_lite/tensorflow/lite/micro/recording_simple_memory_allocator.o \
./tensorflow_lite/tensorflow/lite/micro/simple_memory_allocator.o \
./tensorflow_lite/tensorflow/lite/micro/system_setup.o \
./tensorflow_lite/tensorflow/lite/micro/test_helpers.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow_lite/tensorflow/lite/micro/%.o: ../tensorflow_lite/tensorflow/lite/micro/%.cc tensorflow_lite/tensorflow/lite/micro/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m7 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/flatbuffers/include" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/gemmlowp" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/ruy" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/kissfft" -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

