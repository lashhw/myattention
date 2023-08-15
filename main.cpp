#include <iostream>
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "my_attention.h"

extern const unsigned char my_attention_model_tflite[];

using op_resolver_t = tflite::MicroMutableOpResolver<2>;

TfLiteStatus RegisterOps(op_resolver_t &op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddCustom("MyAttention", tflite::Register_MY_ATTENTION()));
    return kTfLiteOk;
}

constexpr int kTensorArenaSize = 300000;
uint8_t tensor_arena[kTensorArenaSize];

TfLiteStatus LoadFloatModelAndPerformInference() {
    const tflite::Model *model = tflite::GetModel(my_attention_model_tflite);
    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

    op_resolver_t op_resolver;
    TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

    tflite::RecordingMicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

    interpreter.GetMicroAllocator().PrintAllocations();

    TfLiteTensor *input = interpreter.input(0);
    for (int i = 0; i < input->bytes / sizeof(float); i++)
        input->data.f[i] = (float)i;
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    TfLiteTensor *output = interpreter.output(0);
    for (int i = 0; i < output->bytes / sizeof(float); i++)
        std::cout << output->data.f[i] << std::endl;

    return kTfLiteOk;
}

int main(int argc, char* argv[]) {
    tflite::InitializeTarget();
    TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
    MicroPrintf("~~~ALL TESTS PASSED~~~\n");
    return kTfLiteOk;
}
