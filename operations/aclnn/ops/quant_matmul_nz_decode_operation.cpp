/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "operations/aclnn/ops/quant_matmul_nz_decode_operation.h"

#include <cstdlib>
#include <dlfcn.h>

#include <string>

#include "acl/acl.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
namespace {

constexpr char kGetWorkspaceSizeSymbol[] =
    "aclnnQuantMatmulNzDecodeGetWorkspaceSize";
constexpr char kExecuteSymbol[] = "aclnnQuantMatmulNzDecode";

using GetWorkspaceSizeFunction = aclnnStatus (*)(
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, uint64_t*, aclOpExecutor**);
using ExecuteFunction = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*,
                                        aclrtStream);

bool has_custom_api_symbols(void* handle) {
  return handle != nullptr &&
         dlsym(handle, kGetWorkspaceSizeSymbol) != nullptr &&
         dlsym(handle, kExecuteSymbol) != nullptr;
}

void* open_custom_api_library(const std::string& library_path) {
  void* handle = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (has_custom_api_symbols(handle)) {
    return handle;
  }
  if (handle != nullptr) {
    dlclose(handle);
  }
  return nullptr;
}

void* get_custom_api_handle() {
  static void* handle = []() -> void* {
    const char* custom_opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (custom_opp_path != nullptr) {
      const std::string search_paths(custom_opp_path);
      std::size_t begin = 0;
      while (begin <= search_paths.size()) {
        const std::size_t end = search_paths.find(':', begin);
        const std::string vendor_path =
            search_paths.substr(begin, end - begin);
        if (!vendor_path.empty()) {
          void* vendor_handle = open_custom_api_library(
              vendor_path + "/op_api/lib/libcust_opapi.so");
          if (vendor_handle != nullptr) {
            return vendor_handle;
          }
        }
        if (end == std::string::npos) {
          break;
        }
        begin = end + 1;
      }
    }
    return open_custom_api_library("libcust_opapi.so");
  }();
  return handle;
}

GetWorkspaceSizeFunction get_workspace_size_function() {
  static GetWorkspaceSizeFunction function = []() {
    void* handle = get_custom_api_handle();
    return handle == nullptr
               ? nullptr
               : reinterpret_cast<GetWorkspaceSizeFunction>(
                     dlsym(handle, kGetWorkspaceSizeSymbol));
  }();
  return function;
}

ExecuteFunction get_execute_function() {
  static ExecuteFunction function = []() {
    void* handle = get_custom_api_handle();
    return handle == nullptr
               ? nullptr
               : reinterpret_cast<ExecuteFunction>(
                     dlsym(handle, kExecuteSymbol));
  }();
  return function;
}

}  // namespace

QuantMatmulNzDecodeOperation::QuantMatmulNzDecodeOperation(
    const std::string& name)
    : AclNNOperation(name) {}

bool QuantMatmulNzDecodeOperation::is_available() {
  return get_workspace_size_function() != nullptr &&
         get_execute_function() != nullptr;
}

atb::Status QuantMatmulNzDecodeOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& in_tensor_descs,
    atb::SVector<atb::TensorDesc>& out_tensor_descs) const {
  if (in_tensor_descs.size() != GetInputNum() ||
      out_tensor_descs.size() != GetOutputNum()) {
    ATB_SPEED_LOG_ERROR(opName_ << " received an invalid tensor count");
    return atb::ERROR_INVALID_PARAM;
  }

  const atb::TensorDesc& x_desc = in_tensor_descs.at(0);
  const atb::TensorDesc& weight_desc = in_tensor_descs.at(1);
  const atb::TensorDesc& scale_desc = in_tensor_descs.at(2);
  const atb::TensorDesc& bias_desc = in_tensor_descs.at(3);
  if (x_desc.shape.dimNum != NUM2 || weight_desc.shape.dimNum != NUM2 ||
      scale_desc.shape.dimNum != NUM1 || bias_desc.shape.dimNum != NUM1) {
    ATB_SPEED_LOG_ERROR(opName_ << " received an unsupported rank");
    return atb::ERROR_INVALID_PARAM;
  }
  const int64_t k_dim = x_desc.shape.dims[DIM1];
  const int64_t n_dim = weight_desc.shape.dims[DIM1];
  const bool is_gate_up_shape = k_dim == 5120 && n_dim == 6400;
  const bool is_down_shape = k_dim == 3200 && n_dim == 5120;
  const bool is_qkv_shape = k_dim == 5120 && n_dim == 1280;
  if (x_desc.shape.dims[DIM0] <= 0 || x_desc.shape.dims[DIM0] > 16 ||
      (!is_gate_up_shape && !is_down_shape && !is_qkv_shape) ||
      weight_desc.shape.dims[DIM0] != k_dim ||
      scale_desc.shape.dims[DIM0] != n_dim ||
      bias_desc.shape.dims[DIM0] != n_dim) {
    ATB_SPEED_LOG_ERROR(opName_ << " received an unsupported shape");
    return atb::ERROR_INVALID_PARAM;
  }

  out_tensor_descs.at(0).format = ACL_FORMAT_ND;
  out_tensor_descs.at(0).dtype = ACL_BF16;
  out_tensor_descs.at(0).shape = x_desc.shape;
  out_tensor_descs.at(0).shape.dims[DIM1] = n_dim;
  return atb::NO_ERROR;
}

uint32_t QuantMatmulNzDecodeOperation::GetInputNum() const {
  return NUM4;
}

uint32_t QuantMatmulNzDecodeOperation::GetOutputNum() const {
  return NUM1;
}

atb::Dims QuantMatmulNzDecodeOperation::get_weight_storage_shape(
    const atb::TensorDesc& tensor_desc) const {
  atb::Dims storage_shape = tensor_desc.shape;
  if (tensor_desc.format == ACL_FORMAT_FRACTAL_NZ) {
    constexpr int64_t kM0 = 16;
    constexpr int64_t kN0 = 32;
    storage_shape.dimNum = NUM4;
    storage_shape.dims[DIM0] = tensor_desc.shape.dims[DIM1] / kN0;
    storage_shape.dims[DIM1] = tensor_desc.shape.dims[DIM0] / kM0;
    storage_shape.dims[DIM2] = kM0;
    storage_shape.dims[DIM3] = kN0;
  }
  return storage_shape;
}

std::shared_ptr<AclNNTensor> QuantMatmulNzDecodeOperation::create_tensor(
    const atb::Tensor& atb_tensor,
    int32_t tensor_idx,
    bool is_weight) const {
  auto aclnn_tensor = std::make_shared<AclNNTensor>();
  aclnn_tensor->tensorIdx = tensor_idx;
  aclnn_tensor->needUpdateTensorDataPtr = true;
  aclnn_tensor->atbTensor = atb_tensor;
  atb::Dims view_shape = atb_tensor.desc.shape;
  aclnn_tensor->strides = GetCopyTensorStride(view_shape);
  const atb::Dims storage_shape =
      is_weight ? get_weight_storage_shape(atb_tensor.desc)
                : atb_tensor.desc.shape;
  aclnn_tensor->tensor = aclCreateTensor(
      atb_tensor.desc.shape.dims, atb_tensor.desc.shape.dimNum,
      atb_tensor.desc.dtype, aclnn_tensor->strides.data(), 0,
      atb_tensor.desc.format, storage_shape.dims, storage_shape.dimNum,
      atb_tensor.deviceData);
  return aclnn_tensor;
}

int QuantMatmulNzDecodeOperation::CreateAclNNVariantPack(
    const atb::VariantPack& variant_pack) {
  if (variant_pack.inTensors.size() != GetInputNum() ||
      variant_pack.outTensors.size() != GetOutputNum()) {
    return atb::ERROR_INVALID_PARAM;
  }

  AclNNVariantPack& aclnn_variant_pack =
      this->aclnnOpCache_->aclnnVariantPack;
  aclnn_variant_pack.aclInTensors.resize(GetInputNum());
  for (std::size_t i = 0; i < aclnn_variant_pack.aclInTensors.size(); ++i) {
    aclnn_variant_pack.aclInTensors[i] = create_tensor(
        variant_pack.inTensors.at(i), static_cast<int32_t>(i), i == DIM1);
    if (aclnn_variant_pack.aclInTensors[i]->tensor == nullptr) {
      ATB_SPEED_LOG_ERROR(opName_ << " failed to create input tensor " << i);
      return atb::ERROR_INTERNAL_ERROR;
    }
  }

  aclnn_variant_pack.aclOutTensors.resize(GetOutputNum());
  aclnn_variant_pack.aclOutTensors[0] =
      create_tensor(variant_pack.outTensors.at(0), 0, false);
  if (aclnn_variant_pack.aclOutTensors[0]->tensor == nullptr) {
    ATB_SPEED_LOG_ERROR(opName_ << " failed to create output tensor");
    return atb::ERROR_INTERNAL_ERROR;
  }
  return atb::NO_ERROR;
}

int QuantMatmulNzDecodeOperation::SetAclNNWorkspaceExecutor() {
  GetWorkspaceSizeFunction get_workspace_size =
      get_workspace_size_function();
  if (get_workspace_size == nullptr) {
    ATB_SPEED_LOG_ERROR(opName_ << " custom operator API is unavailable");
    return atb::ERROR_INTERNAL_ERROR;
  }

  AclNNVariantPack& pack = this->aclnnOpCache_->aclnnVariantPack;
  return get_workspace_size(
      pack.aclInTensors.at(0)->tensor, pack.aclInTensors.at(1)->tensor,
      pack.aclInTensors.at(2)->tensor, pack.aclInTensors.at(3)->tensor,
      pack.aclOutTensors.at(0)->tensor,
      &this->aclnnOpCache_->workspaceSize,
      &this->aclnnOpCache_->aclExecutor);
}

int QuantMatmulNzDecodeOperation::ExecuteAclNNOp(
    uint8_t* workspace,
    aclrtStream& stream) {
  ExecuteFunction execute = get_execute_function();
  if (execute == nullptr) {
    ATB_SPEED_LOG_ERROR(opName_ << " custom operator execute API is unavailable");
    return atb::ERROR_INTERNAL_ERROR;
  }
  return execute(workspace, this->aclnnOpCache_->workspaceSize,
                 this->aclnnOpCache_->aclExecutor, stream);
}

}  // namespace common
}  // namespace atb_speed
