################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tensorflow_lite/tensorflow/lite/core/api/error_reporter.cc \
../tensorflow_lite/tensorflow/lite/core/api/flatbuffer_conversions.cc \
../tensorflow_lite/tensorflow/lite/core/api/op_resolver.cc \
../tensorflow_lite/tensorflow/lite/core/api/tensor_utils.cc 

CC_DEPS += \
./tensorflow_lite/tensorflow/lite/core/api/error_reporter.d \
./tensorflow_lite/tensorflow/lite/core/api/flatbuffer_conversions.d \
./tensorflow_lite/tensorflow/lite/core/api/op_resolver.d \
./tensorflow_lite/tensorflow/lite/core/api/tensor_utils.d 

OBJS += \
./tensorflow_lite/tensorflow/lite/core/api/error_reporter.o \
./tensorflow_lite/tensorflow/lite/core/api/flatbuffer_conversions.o \
./tensorflow_lite/tensorflow/lite/core/api/op_resolver.o \
./tensorflow_lite/tensorflow/lite/core/api/tensor_utils.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow_lite/tensorflow/lite/core/api/%.o: ../tensorflow_lite/tensorflow/lite/core/api/%.cc tensorflow_lite/tensorflow/lite/core/api/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m7 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/flatbuffers/include" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/gemmlowp" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/ruy" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/kissfft" -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

