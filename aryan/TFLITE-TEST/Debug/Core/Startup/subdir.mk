################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Core/Startup/startup_stm32f746nghx.s 

S_DEPS += \
./Core/Startup/startup_stm32f746nghx.d 

OBJS += \
./Core/Startup/startup_stm32f746nghx.o 


# Each subdirectory must supply rules for building sources it contributes
Core/Startup/%.o: ../Core/Startup/%.s Core/Startup/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m7 -g3 -DDEBUG -c -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/flatbuffers/include" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/gemmlowp" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/ruy" -I"/home/aryan/Desktop/DDP/TFLITE-TEST/tensorflow_lite/third_party/kissfft" -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

