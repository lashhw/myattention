#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

struct OpData {
    int num_heads;
    int key_dim;
    int scratch_index;
};

struct ShapeInfo {
    int query_batches;
    int query_features;
    int value_batches;
    int value_features;
};

ShapeInfo GetShapeInfo(TfLiteIntArray *query_dims, TfLiteIntArray *value_dims) {
    ShapeInfo shape_info{};
    int query_dims_count = query_dims->size;
    int value_dims_count = value_dims->size;
    TFLITE_DCHECK_GE(query_dims_count, 2);
    TFLITE_DCHECK_GE(value_dims_count, 2);

    shape_info.query_features = query_dims->data[query_dims_count - 1];
    shape_info.value_features = value_dims->data[value_dims_count - 1];
    shape_info.query_batches = 1;
    shape_info.value_batches = 1;
    for (int i = 0; i < query_dims_count - 1; i++)
        shape_info.query_batches *= query_dims->data[i];
    for (int i = 0; i < value_dims_count - 1; i++)
        shape_info.value_batches *= value_dims->data[i];
    return shape_info;
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
    TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
    auto *op_data = static_cast<OpData*>(
        context->AllocatePersistentBuffer(context, sizeof(OpData)));
    TFLITE_DCHECK(op_data != nullptr);

    // get parameters, the values are ordered alphabetically
    TFLITE_DCHECK(buffer != nullptr);
    TFLITE_DCHECK(length > 0);
    tflite::FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer), length);
    op_data->key_dim = fbw.ElementAsInt32(0);
    op_data->num_heads = fbw.ElementAsInt32(1);

    return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
    MicroContext *micro_context = GetMicroContext(context);

    TF_LITE_ENSURE(context, node->user_data != nullptr);
    auto *op_data = reinterpret_cast<OpData*>(node->user_data);

    TfLiteTensor *query = micro_context->AllocateTempInputTensor(node, 0);
    TfLiteTensor *value = micro_context->AllocateTempInputTensor(node, 1);
    TfLiteTensor *query_kernel = micro_context->AllocateTempInputTensor(node, 2);
    TfLiteTensor *query_bias = micro_context->AllocateTempInputTensor(node, 3);
    TfLiteTensor *key_kernel = micro_context->AllocateTempInputTensor(node, 4);
    TfLiteTensor *key_bias = micro_context->AllocateTempInputTensor(node, 5);
    TfLiteTensor *value_kernel = micro_context->AllocateTempInputTensor(node, 6);
    TfLiteTensor *value_bias = micro_context->AllocateTempInputTensor(node, 7);
    TfLiteTensor *output_kernel = micro_context->AllocateTempInputTensor(node, 8);
    TfLiteTensor *output_bias = micro_context->AllocateTempInputTensor(node, 9);
    TfLiteTensor *attention_output = micro_context->AllocateTempOutputTensor(node, 0);

    ShapeInfo shape_info = GetShapeInfo(query->dims, value->dims);

    TF_LITE_ENSURE_EQ(context, query_kernel->dims->size, 3);
    TF_LITE_ENSURE_EQ(context, query_kernel->dims->data[0], shape_info.query_features);
    TF_LITE_ENSURE_EQ(context, query_kernel->dims->data[1], op_data->num_heads);
    TF_LITE_ENSURE_EQ(context, query_kernel->dims->data[2], op_data->key_dim);

    TF_LITE_ENSURE_EQ(context, query_bias->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, query_bias->dims->data[0], op_data->num_heads);
    TF_LITE_ENSURE_EQ(context, query_bias->dims->data[1], op_data->key_dim);

    TF_LITE_ENSURE_EQ(context, key_kernel->dims->size, 3);
    TF_LITE_ENSURE_EQ(context, key_kernel->dims->data[0], shape_info.value_features);
    TF_LITE_ENSURE_EQ(context, key_kernel->dims->data[1], op_data->num_heads);
    TF_LITE_ENSURE_EQ(context, key_kernel->dims->data[2], op_data->key_dim);

    TF_LITE_ENSURE_EQ(context, key_bias->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, key_bias->dims->data[0], op_data->num_heads);
    TF_LITE_ENSURE_EQ(context, key_bias->dims->data[1], op_data->key_dim);

    TF_LITE_ENSURE_EQ(context, value_kernel->dims->size, 3);
    TF_LITE_ENSURE_EQ(context, value_kernel->dims->data[0], shape_info.value_features);
    TF_LITE_ENSURE_EQ(context, value_kernel->dims->data[1], op_data->num_heads);
    TF_LITE_ENSURE_EQ(context, value_kernel->dims->data[2], op_data->key_dim);

    TF_LITE_ENSURE_EQ(context, value_bias->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, value_bias->dims->data[0], op_data->num_heads);
    TF_LITE_ENSURE_EQ(context, value_bias->dims->data[1], op_data->key_dim);

    TF_LITE_ENSURE_EQ(context, output_kernel->dims->size, 3);
    TF_LITE_ENSURE_EQ(context, output_kernel->dims->data[0], op_data->num_heads);
    TF_LITE_ENSURE_EQ(context, output_kernel->dims->data[1], op_data->key_dim);
    TF_LITE_ENSURE_EQ(context, output_kernel->dims->data[2], shape_info.query_features);

    TF_LITE_ENSURE_EQ(context, output_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, output_bias->dims->data[0], shape_info.query_features);

    size_t scratch_size = 2 * shape_info.value_batches * op_data->key_dim /* kproj and vproj */ +
                          op_data->key_dim /* qbuf */ + op_data->key_dim /* vbuf */;
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(context, scratch_size, &op_data->scratch_index));

    micro_context->DeallocateTempTfLiteTensor(query);
    micro_context->DeallocateTempTfLiteTensor(value);
    micro_context->DeallocateTempTfLiteTensor(query_kernel);
    micro_context->DeallocateTempTfLiteTensor(query_bias);
    micro_context->DeallocateTempTfLiteTensor(key_kernel);
    micro_context->DeallocateTempTfLiteTensor(key_bias);
    micro_context->DeallocateTempTfLiteTensor(value_kernel);
    micro_context->DeallocateTempTfLiteTensor(value_bias);
    micro_context->DeallocateTempTfLiteTensor(output_kernel);
    micro_context->DeallocateTempTfLiteTensor(output_bias);
    micro_context->DeallocateTempTfLiteTensor(attention_output);

    return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext *context, TfLiteNode *node) {
    const TfLiteEvalTensor *query = tflite::micro::GetEvalInput(context, node, 0);
    const TfLiteEvalTensor *value = tflite::micro::GetEvalInput(context, node, 1);
    const TfLiteEvalTensor *query_kernel = tflite::micro::GetEvalInput(context, node, 2);
    const TfLiteEvalTensor *query_bias = tflite::micro::GetEvalInput(context, node, 3);
    const TfLiteEvalTensor *key_kernel = tflite::micro::GetEvalInput(context, node, 4);
    const TfLiteEvalTensor *key_bias = tflite::micro::GetEvalInput(context, node, 5);
    const TfLiteEvalTensor *value_kernel = tflite::micro::GetEvalInput(context, node, 6);
    const TfLiteEvalTensor *value_bias = tflite::micro::GetEvalInput(context, node, 7);
    const TfLiteEvalTensor *output_kernel = tflite::micro::GetEvalInput(context, node, 8);
    const TfLiteEvalTensor *output_bias = tflite::micro::GetEvalInput(context, node, 9);
    TfLiteEvalTensor *attention_output = tflite::micro::GetEvalOutput(context, node, 0);

    TF_LITE_ENSURE(context, node->user_data != nullptr);
    auto *op_data = reinterpret_cast<OpData*>(node->user_data);
    ShapeInfo shape_info = GetShapeInfo(query->dims, value->dims);

    auto *scratch_buffer = reinterpret_cast<float*>(context->GetScratchBuffer(context, op_data->scratch_index));
    float *kproj = scratch_buffer;
    float *vproj = kproj + shape_info.value_batches * op_data->key_dim;
    float *qbuf = vproj + shape_info.value_batches * op_data->key_dim;
    float *vbuf = qbuf + op_data->key_dim;

    auto calculate_proj = [&](int h, const float *in_kernel, const float *in_bias, float *out) {
        for (int b = 0; b < shape_info.value_batches; b++) {
            for (int k = 0; k < op_data->key_dim; k++) {
                float sum = in_bias[h * op_data->key_dim + k];
                for (int f = 0; f < shape_info.value_features; f++)
                    sum += value->data.f[b * shape_info.value_features + f] *
                           in_kernel[f * op_data->num_heads * op_data->key_dim + h * op_data->key_dim + k];
                out[b * op_data->key_dim + k] = sum;
            }
        }
    };

    float inv_sqrt = 1.0f / sqrtf(static_cast<float>(op_data->key_dim));
    for (int b = 0; b < shape_info.query_batches; b++)
        for (int f = 0; f < shape_info.query_features; f++)
            attention_output->data.f[b * shape_info.query_features + f] = output_bias->data.f[f];

    for (int h = 0; h < op_data->num_heads; h++) {
        calculate_proj(h, key_kernel->data.f, key_bias->data.f, kproj);
        calculate_proj(h, value_kernel->data.f, value_bias->data.f, vproj);

        for (int qb = 0; qb < shape_info.query_batches; qb++) {
            for (int k = 0; k < op_data->key_dim; k++) {
                float sum = query_bias->data.f[h * op_data->key_dim + k];
                for (int f = 0; f < shape_info.query_features; f++)
                    sum += query->data.f[qb * shape_info.query_features + f] *
                           query_kernel->data.f[f * op_data->num_heads * op_data->key_dim + h * op_data->key_dim + k];
                qbuf[k] = sum;
            }

            float exp_sum = 0.0f;
            for (int k = 0; k < op_data->key_dim; k++)
                vbuf[k] = 0.0f;

            for (int vb = 0; vb < shape_info.value_batches; vb++) {
                float qk = 0.0f;
                for (int k = 0; k < op_data->key_dim; k++)
                    qk += qbuf[k] * kproj[vb * op_data->key_dim + k];
                qk *= inv_sqrt;
                qk = expf(qk);
                exp_sum += qk;

                for (int k = 0; k < op_data->key_dim; k++)
                    vbuf[k] += qk * vproj[vb * op_data->key_dim + k];
            }

            float inv_exp_sum = 1.0f / exp_sum;
            for (int k = 0; k < op_data->key_dim; k++)
                vbuf[k] *= inv_exp_sum;

            for (int f = 0; f < shape_info.query_features; f++) {
                float sum = 0.0f;
                for (int k = 0; k < op_data->key_dim; k++)
                    sum += vbuf[k] * output_kernel->data.f[h * op_data->key_dim * shape_info.query_features +
                                                           k * shape_info.query_features + f];
                attention_output->data.f[qb * shape_info.query_features + f] += sum;
            }
        }
    }

    return kTfLiteOk;
}

TFLMRegistration *Register_MY_ATTENTION() {
    static TFLMRegistration r = tflite::micro::RegisterOp(Init, Prepare, Invoke);
    return &r;
}

}  // namespace
}  // namespace tflite