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

#pragma once

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

class QuantMatmulNzDecodeOperation final : public AclNNOperation {
 public:
  explicit QuantMatmulNzDecodeOperation(const std::string& name);
  ~QuantMatmulNzDecodeOperation() override = default;
  static bool is_available();

  atb::Status InferShape(
      const atb::SVector<atb::TensorDesc>& in_tensor_descs,
      atb::SVector<atb::TensorDesc>& out_tensor_descs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;

 protected:
  int CreateAclNNVariantPack(const atb::VariantPack& variant_pack) override;
  int SetAclNNWorkspaceExecutor() override;
  int ExecuteAclNNOp(uint8_t* workspace, aclrtStream& stream) override;

 private:
  std::shared_ptr<AclNNTensor> create_tensor(const atb::Tensor& atb_tensor,
                                             int32_t tensor_idx,
                                             bool is_weight) const;
  atb::Dims get_weight_storage_shape(
      const atb::TensorDesc& tensor_desc) const;
};

}  // namespace common
}  // namespace atb_speed
