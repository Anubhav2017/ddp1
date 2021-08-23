################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../tensorflow_lite/tensorflow/lite/c/common.c 

C_DEPS += \
./tensorflow_lite/tensorflow/lite/c/common.d 

OBJS += \
./tensorflow_lite/tensorflow/lite/c/common.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow_lite/tensorflow/lite/c/%.o: ../tensorflow_lite/tensorflow/lite/c/%.c tensorflow_lite/tensorflow/lite/c/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/flatbuffers/include" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/gemmlowp" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/ruy" -I"C:/Users/agarw/Documents/ddp/aryan/TFLITE-TEST/tensorflow_lite/third_party/kissfft" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

