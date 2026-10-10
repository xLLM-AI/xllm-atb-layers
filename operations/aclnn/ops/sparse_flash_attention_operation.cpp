/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "sparse_flash_attention_operation.h"
#include "operations/aclnn/utils/op_api_resolver.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
namespace {

// The 23-param aclnnSparseFlashAttention API shipped with CANN 9.0.0/9.1.0.
using AclnnSparseFlashAttentionGetWorkspaceSizeFunc = aclnnStatus (*)(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    double scaleValue,
    int64_t sparseBlockSizeOptional,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    bool returnSoftmaxLse,
    const aclTensor *attentionOut,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
using AclnnSparseFlashAttentionFunc =
    aclnnStatus (*)(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

} // namespace

SparseFlashAttentionOperation::SparseFlashAttentionOperation(const std::string &name,
                                                                 SparseFlashAttentionParam param)
    : AclNNOperation(name), param_(param)
{
}

SparseFlashAttentionOperation::~SparseFlashAttentionOperation()
{
    ATB_SPEED_LOG_DEBUG("SparseFlashAttentionOperation deconstruct");
    this->DestroyOperation();
}

uint32_t SparseFlashAttentionOperation::GetInputNum() const
{
    return 9; // query, key, weight, seq_len_query, seq_len_key, block_table
}

uint32_t SparseFlashAttentionOperation::GetOutputNum() const { return 1; }

atb::Status SparseFlashAttentionOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                                        atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SparseFlashAttentionOperation infer shape start");

    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
    if (param_.queryLayout == "TND") {
        outTensorDesc.at(0).shape.dimNum = 3;
        outTensorDesc.at(0).shape.dims[0] = inTensorDesc.at(0).shape.dims[0];
        outTensorDesc.at(0).shape.dims[1] = inTensorDesc.at(0).shape.dims[1];
        outTensorDesc.at(0).shape.dims[2] = inTensorDesc.at(0).shape.dims[2];
    } else {
        outTensorDesc.at(0).shape.dimNum = 4;
        outTensorDesc.at(0).shape.dims[0] = inTensorDesc.at(0).shape.dims[0];
        outTensorDesc.at(0).shape.dims[1] = inTensorDesc.at(0).shape.dims[1];
        outTensorDesc.at(0).shape.dims[2] = inTensorDesc.at(0).shape.dims[2];
        outTensorDesc.at(0).shape.dims[3] = inTensorDesc.at(0).shape.dims[3];
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "SparseFlashAttentionOperation InferShape end");

    return atb::NO_ERROR;
}

atb::Status SparseFlashAttentionOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        if (!param_.hasBlockTable && i == 4) {  // 4: index of block_table
            std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
            aclnnVariantPack.aclInTensors[i] = aclnnTensor;
            continue;
        }
        aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), i);
        if (aclnnVariantPack.aclInTensors[i]->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

atb::Status SparseFlashAttentionOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int SparseFlashAttentionOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    // Resolve the op-api entry from the vendor op-api libraries instead of
    // calling the CANN built-in entry: the op definition keeps evolving in the
    // ops-transformer vendor package (e.g. the sinks input added after CANN
    // 9.1.0), so the built-in wrapper marshals parameters per an older
    // definition than the vendor opmaster/kernel selected by the aclInit
    // registry, producing silently wrong results.
    // CANN 9.0's SparseFlashAttention adds 4 inputs (fixed defaults) and 2
    // outputs. See:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/aolapi/context/ops-transformer/aclnnSparseFlashAttention.md
    static const auto aclnnSparseFlashAttentionGetWorkspaceSize =
        GetOpApiFunc<AclnnSparseFlashAttentionGetWorkspaceSizeFunc>("aclnnSparseFlashAttentionGetWorkspaceSize");
    if (aclnnSparseFlashAttentionGetWorkspaceSize == nullptr) {
        return kAclnnOpApiNotFound;
    }
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    int64_t attentionMode = 2;
    bool returnSoftmaxLse = false;
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnSparseFlashAttentionGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor, // query
        aclnnVariantPack.aclInTensors.at(1)->tensor, // key
        aclnnVariantPack.aclInTensors.at(2)->tensor, // value
        aclnnVariantPack.aclInTensors.at(3)->tensor, // sparse_indices
        param_.hasBlockTable ? aclnnVariantPack.aclInTensors.at(4)->tensor : nullptr, // block_table
        aclnnVariantPack.aclInTensors.at(5)->tensor, // actual_seq_lenths_query
        aclnnVariantPack.aclInTensors.at(6)->tensor, // actual_seq_lenths_kv,
        aclnnVariantPack.aclInTensors.at(7)->tensor, // query_rope,
        aclnnVariantPack.aclInTensors.at(8)->tensor, // key_rope
        param_.scaleValue, param_.sparseBlockSize,
        const_cast<char *>(param_.queryLayout.c_str()), // query_layout
        const_cast<char *>(param_.kvLayout.c_str()),   // key_layout
        param_.sparseMode,
        preTokens, nextTokens, attentionMode, returnSoftmaxLse, // cann9.0 add
        aclnnVariantPack.aclOutTensors.at(0)->tensor, // out
        nullptr, // softmax_max, unused since returnSoftmaxLse=false
        nullptr, // softmax_sum, unused since returnSoftmaxLse=false
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                                << ", ret: " << ret << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                                << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int SparseFlashAttentionOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    static const auto aclnnSparseFlashAttention =
        GetOpApiFunc<AclnnSparseFlashAttentionFunc>("aclnnSparseFlashAttention");
    if (aclnnSparseFlashAttention == nullptr) {
        return kAclnnOpApiNotFound;
    }
    int ret = aclnnSparseFlashAttention(workspace, this->aclnnOpCache_->workspaceSize,
                                          this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed
