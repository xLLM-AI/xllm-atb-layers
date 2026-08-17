/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

struct MegaMoeParam {
    int64_t moeExpertNum = 1;
    int64_t epWorldSize = 1;
    int64_t cclBufferSize = 0;
    int64_t maxRecvTokenNum = 0;
    int64_t dispatchQuantMode = 0;
    int64_t dispatchQuantOutDtype = ACL_DT_UNDEFINED;
    int64_t combineQuantMode = 0;
    int64_t numMaxTokensPerRank = 0;
    std::string commAlg = "";
    std::string activation = "swiglu";
    float activationClamp = std::numeric_limits<float>::max();

    uint32_t weight1TensorNum = 1;
    uint32_t weight2TensorNum = 1;
    uint32_t weightScales1TensorNum = 0;
    uint32_t weightScales2TensorNum = 0;
    uint32_t bias1TensorNum = 0;
    uint32_t bias2TensorNum = 0;
    bool hasXActiveMask = false;
};

uint32_t GetMegaMoeInputNum(const MegaMoeParam &param);

class MegaMoeOperation final : public AclNNOperation {
public:
    explicit MegaMoeOperation(const std::string &name, MegaMoeParam param);
    ~MegaMoeOperation() override;

    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

    atb::Status AppendAclNNInputTensorList(const atb::VariantPack &variantPack,
                                           uint32_t tensorCount,
                                           uint32_t listType,
                                           uint32_t &atbTensorIdx);
    aclTensorList *GetAclInTensorList(const AclNNVariantPack &aclnnVariantPack, uint32_t listType) const;

    MegaMoeParam param_;
    std::vector<std::vector<std::shared_ptr<AclNNTensor>>> inputTensorObjects_;
    std::vector<std::vector<aclTensor *>> inputTensorVectors_;
    std::vector<int32_t> inputTensorListIdx_;
};

}  // namespace common
}  // namespace atb_speed
