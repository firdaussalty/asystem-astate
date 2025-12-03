#pragma once

#include <string>

#include <cuda_runtime.h>
#include <memory.h>

#include "core/atensor.h"

namespace astate {

// H2H serialize & deserialize
#define SERIALIZE_FIELD_H2H(dest, src, size) \
    memcpy(dest, src, size); \
    dest += size;

#define DESERIALIZE_FIELD_H2H(dest, src, size) \
    memcpy(dest, src, size); \
    src += size;

// D2H, D2D, H2D serialize & deserialize
#define SERIALIZE_FIELD_D2X_X2D(dest, src, size, kind) \
    cudaMemcpy(dest, src, size, kind); \
    dest += size;

#define DESERIALIZE_FIELD_D2X_X2D(dest, src, size, kind) \
    cudaMemcpy(dest, src, size, kind); \
    src += size;

size_t SerializeTensorMeta(const ATensor& atensor, void* buffer);
std::shared_ptr<ATensor> DeserializeTensorMeta(void* buffer);

std::string SerializeTensorMeta(const ATensor& atensor);
std::shared_ptr<ATensor> DeserializeTensorMeta(const std::string& data);

size_t SerializeTensorData(const ATensor& atensor, void* buffer, bool to_device = false);
void DeserializeTensorData(ATensor& atensor, void* buffer, bool from_device = false);
} // namespace astate
