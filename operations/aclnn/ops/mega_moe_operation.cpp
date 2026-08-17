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
#include "operations/aclnn/ops/mega_moe_operation.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
namespace {

constexpr uint32_t kMegaMoeRequiredTensorNum = 4;
constexpr uint32_t kMegaMoeTensorListNum = 6;
constexpr uint32_t kAclContextTensorIdx = 0;
constexpr uint32_t kAclXTensorIdx = 1;
constexpr uint32_t kAclTopkIdsTensorIdx = 2;
constexpr uint32_t kAclTopkWeightsTensorIdx = 3;
constexpr uint32_t kAclXActiveMaskTensorIdx = 10;
constexpr uint32_t kMegaMoeOutputNum = 2;
constexpr uint32_t kWeight1TensorListIdx = 0;
constexpr uint32_t kWeight2TensorListIdx = 1;
constexpr uint32_t kWeightScales1TensorListIdx = 2;
constexpr uint32_t kWeightScales2TensorListIdx = 3;
constexpr uint32_t kBias1TensorListIdx = 4;
constexpr uint32_t kBias2TensorListIdx = 5;
constexpr int64_t kCubeBlockSize = 16;
constexpr int64_t kBf16NzBlockSize = 32;
constexpr int64_t kInt4NzBlockSize = 32;
constexpr int64_t kInt4PackedBlockSize = 64;
constexpr int kMegaMoeAclnnNotFound = -1;

using AclnnMegaMoeGetWorkspaceSizeFunc = aclnnStatus (*)(
    const aclTensor *context,
    const aclTensor *x,
    const aclTensor *topkIds,
    const aclTensor *topkWeights,
    const aclTensorList *weight1,
    const aclTensorList *weight2,
    const aclTensorList *weightScales1Optional,
    const aclTensorList *weightScales2Optional,
    const aclTensorList *bias1Optional,
    const aclTensorList *bias2Optional,
    const aclTensor *xActiveMaskOptional,
    int64_t moeExpertNum,
    int64_t epWorldSize,
    int64_t cclBufferSize,
    int64_t maxRecvTokenNum,
    int64_t dispatchQuantMode,
    int64_t dispatchQuantOutDtype,
    int64_t combineQuantMode,
    const char *commAlg,
    int64_t numMaxTokensPerRank,
    const char *activation,
    float activationClamp,
    aclTensor *yOut,
    aclTensor *expertTokenNumsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
using AclnnMegaMoeFunc = aclnnStatus (*)(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream);

struct OpApiLib {
    std::string path;
    void *handle = nullptr;
};

std::vector<std::string> SplitString(const std::string &value, char delimiter)
{
    std::vector<std::string> result;
    std::string token;
    for (char ch : value) {
        if (ch == delimiter) {
            if (!token.empty()) {
                result.push_back(token);
            }
            token.clear();
            continue;
        }
        token.push_back(ch);
    }
    if (!token.empty()) {
        result.push_back(token);
    }
    return result;
}

void AppendCustomOppLibs(std::vector<std::string> &libPaths)
{
    const char *customOppPath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (customOppPath == nullptr) {
        return;
    }

    for (const std::string &path : SplitString(customOppPath, ':')) {
        libPaths.push_back(path + "/op_api/lib/libcust_opapi.so");
    }
}

void AppendDefaultVendorLibs(std::vector<std::string> &libPaths)
{
    const char *oppPath = std::getenv("ASCEND_OPP_PATH");
    if (oppPath == nullptr) {
        return;
    }

    const std::string vendorsPath = std::string(oppPath) + "/vendors";
    std::ifstream configFile(vendorsPath + "/config.ini");
    std::string line;
    while (std::getline(configFile, line)) {
        const std::string loadPriorityPrefix = "load_priority=";
        if (line.find(loadPriorityPrefix) != 0) {
            continue;
        }
        line.erase(0, loadPriorityPrefix.size());
        for (const std::string &vendor : SplitString(line, ',')) {
            libPaths.push_back(vendorsPath + "/" + vendor + "/op_api/lib/libcust_opapi.so");
        }
        break;
    }
}

std::vector<OpApiLib> OpenOpApiLibs()
{
    std::vector<std::string> libPaths;
    AppendCustomOppLibs(libPaths);
    AppendDefaultVendorLibs(libPaths);
    libPaths.push_back("libcust_opapi.so");
    libPaths.push_back("libopapi.so");

    std::vector<OpApiLib> libs;
    libs.reserve(libPaths.size());
    for (const std::string &libPath : libPaths) {
        void *handle = dlopen(libPath.c_str(), RTLD_LAZY);
        if (handle == nullptr) {
            ATB_SPEED_LOG_DEBUG("MegaMoe dlopen " << libPath << " failed, error:" << dlerror());
            continue;
        }
        libs.push_back({libPath, handle});
    }
    return libs;
}

void *GetOpApiFuncAddr(const char *apiName)
{
    static const std::vector<OpApiLib> libs = OpenOpApiLibs();
    for (const OpApiLib &lib : libs) {
        dlerror();
        void *funcAddr = dlsym(lib.handle, apiName);
        const char *error = dlerror();
        if (error == nullptr && funcAddr != nullptr) {
            ATB_SPEED_LOG_DEBUG("MegaMoe found " << apiName << " in " << lib.path);
            return funcAddr;
        }
    }
    ATB_SPEED_LOG_ERROR("MegaMoe can not find " << apiName << " in opapi libs");
    return nullptr;
}

int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    return (dividend + divisor - 1) / divisor;
}

bool IsBf16(const atb::TensorDesc &tensorDesc)
{
    return tensorDesc.dtype == ACL_BF16;
}

bool IsInt4(const atb::TensorDesc &tensorDesc)
{
    return tensorDesc.dtype == ACL_INT4;
}

atb::Dims GetMegaMoeStorageShape(const atb::TensorDesc &tensorDesc)
{
    atb::Dims storageShape = tensorDesc.shape;
    if (tensorDesc.format != ACL_FORMAT_FRACTAL_NZ || tensorDesc.shape.dimNum < 2 ||
        tensorDesc.shape.dimNum + 2 > 8) {
        return storageShape;
    }

    const uint64_t dimNum = tensorDesc.shape.dimNum;
    const uint64_t mDim = dimNum - 2;
    const uint64_t nDim = dimNum - 1;
    const int64_t nBlockSize = IsBf16(tensorDesc) ? kBf16NzBlockSize : kCubeBlockSize;

    storageShape.dimNum = dimNum + 2;
    for (uint64_t i = 0; i < mDim; ++i) {
        storageShape.dims[i] = tensorDesc.shape.dims[i];
    }
    if (IsInt4(tensorDesc)) {
        storageShape.dims[mDim] = CeilDiv(tensorDesc.shape.dims[nDim], kInt4PackedBlockSize);
        storageShape.dims[mDim + 1] = CeilDiv(tensorDesc.shape.dims[mDim], kCubeBlockSize);
        storageShape.dims[mDim + 2] = kCubeBlockSize;
        storageShape.dims[mDim + 3] = kInt4NzBlockSize;
        return storageShape;
    }

    storageShape.dims[mDim] = CeilDiv(tensorDesc.shape.dims[mDim], kCubeBlockSize);
    storageShape.dims[mDim + 1] = CeilDiv(tensorDesc.shape.dims[nDim], nBlockSize);
    storageShape.dims[mDim + 2] = kCubeBlockSize;
    storageShape.dims[mDim + 3] = nBlockSize;
    return storageShape;
}

std::shared_ptr<AclNNTensor> CreateMegaMoeTensor(
    atb::Tensor atbTensor, int32_t tensorListIdx, int32_t tensorIdx, bool useUint64Storage)
{
    if (useUint64Storage) {
        if (atbTensor.desc.dtype != ACL_INT64) {
            ATB_SPEED_LOG_ERROR("MegaMoe encoded weight scale must use INT64 storage, got "
                << atbTensor.desc.dtype);
            return nullptr;
        }
        atbTensor.desc.dtype = ACL_UINT64;
    }
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorListidx = tensorListIdx;
    aclnnTensor->tensorIdx = tensorIdx;
    atb::Dims viewShape = atbTensor.desc.shape;
    atb::Dims storageShape = GetMegaMoeStorageShape(atbTensor.desc);
    aclnnTensor->strides = GetCopyTensorStride(viewShape);
    CallAclCreateTensor(viewShape, storageShape, atbTensor, aclnnTensor);
    return aclnnTensor;
}

std::shared_ptr<AclNNTensor> CreatePackedMegaMoeTensorView(
    const atb::Tensor &packedTensor,
    int32_t tensorListIdx,
    int32_t tensorIdx,
    int64_t expertIdx,
    bool useUint64Storage)
{
    const int64_t expertCount = packedTensor.desc.shape.dims[0];
    atb::Tensor expertTensor = packedTensor;
    expertTensor.desc.shape.dimNum = packedTensor.desc.shape.dimNum - 1;
    for (uint64_t dimIdx = 0; dimIdx < expertTensor.desc.shape.dimNum; ++dimIdx) {
        expertTensor.desc.shape.dims[dimIdx] = packedTensor.desc.shape.dims[dimIdx + 1];
    }
    if (useUint64Storage && expertTensor.desc.shape.dimNum == 2 &&
        expertTensor.desc.shape.dims[1] == 1) {
        expertTensor.desc.shape.dimNum = 1;
    }
    expertTensor.dataSize = packedTensor.dataSize / expertCount;
    expertTensor.deviceData = static_cast<uint8_t *>(packedTensor.deviceData) +
        expertIdx * expertTensor.dataSize;
    std::shared_ptr<AclNNTensor> aclnnTensor = CreateMegaMoeTensor(
        expertTensor, tensorListIdx, tensorIdx, useUint64Storage);
    if (aclnnTensor != nullptr) {
        aclnnTensor->needUpdateTensorDataPtr = false;
    }
    return aclnnTensor;
}

}  // namespace

uint32_t GetMegaMoeInputNum(const MegaMoeParam &param)
{
    uint32_t inputNum = kMegaMoeRequiredTensorNum;
    inputNum += param.weight1TensorNum;
    inputNum += param.weight2TensorNum;
    inputNum += param.weightScales1TensorNum;
    inputNum += param.weightScales2TensorNum;
    inputNum += param.bias1TensorNum;
    inputNum += param.bias2TensorNum;
    if (param.hasXActiveMask) {
        inputNum += 1;
    }
    return inputNum;
}

MegaMoeOperation::MegaMoeOperation(const std::string &name, MegaMoeParam param)
    : AclNNOperation(name), param_(param)
{
}

MegaMoeOperation::~MegaMoeOperation() {}

atb::Status MegaMoeOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MegaMoeOperation infer shape start");
    if (param_.epWorldSize <= 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " epWorldSize must be greater than 0, got " << param_.epWorldSize);
        return atb::ERROR_INVALID_PARAM;
    }

    outTensorDescs.at(DIM0) = inTensorDescs.at(kAclXTensorIdx);

    outTensorDescs.at(DIM1).format = ACL_FORMAT_ND;
    outTensorDescs.at(DIM1).dtype = ACL_INT32;
    outTensorDescs.at(DIM1).shape.dimNum = 1;
    outTensorDescs.at(DIM1).shape.dims[DIM0] = param_.moeExpertNum / param_.epWorldSize;
    ATB_SPEED_LOG_DEBUG(opName_ << " MegaMoeOperation infer shape end");
    return atb::NO_ERROR;
}

uint32_t MegaMoeOperation::GetInputNum() const
{
    return GetMegaMoeInputNum(param_);
}

uint32_t MegaMoeOperation::GetOutputNum() const
{
    return kMegaMoeOutputNum;
}

atb::Status MegaMoeOperation::AppendAclNNInputTensorList(
    const atb::VariantPack &variantPack,
    uint32_t tensorCount,
    uint32_t listType,
    uint32_t &atbTensorIdx)
{
    if (listType >= inputTensorVectors_.size()) {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid TensorList index " << listType);
        return atb::ERROR_INVALID_PARAM;
    }
    inputTensorVectors_.at(listType).clear();
    inputTensorObjects_.at(listType).clear();

    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    if (tensorCount == 0) {
        inputTensorListIdx_.at(listType) = -1;
        aclnnVariantPack.aclInTensorList.push_back(nullptr);
        return atb::NO_ERROR;
    }

    inputTensorListIdx_.at(listType) = static_cast<int32_t>(aclnnVariantPack.aclInTensorList.size());
    for (uint32_t tensorIdx = 0; tensorIdx < tensorCount; ++tensorIdx) {
        const bool useUint64Storage = listType == kWeightScales1TensorListIdx ||
            listType == kWeightScales2TensorListIdx;
        const atb::Tensor &atbTensor = variantPack.inTensors.at(atbTensorIdx);
        const int64_t packedTensorCount = atbTensor.desc.shape.dimNum == 3 ?
            atbTensor.desc.shape.dims[0] : 1;
        for (int64_t packedTensorIdx = 0; packedTensorIdx < packedTensorCount; ++packedTensorIdx) {
            const int32_t aclTensorIdx = static_cast<int32_t>(inputTensorVectors_.at(listType).size());
            std::shared_ptr<AclNNTensor> aclnnTensor = packedTensorCount > 1 ?
                CreatePackedMegaMoeTensorView(atbTensor,
                    static_cast<int32_t>(listType), aclTensorIdx, packedTensorIdx, useUint64Storage) :
                CreateMegaMoeTensor(atbTensor,
                    static_cast<int32_t>(listType), aclTensorIdx, useUint64Storage);
            if (aclnnTensor == nullptr || aclnnTensor->tensor == nullptr) {
                return atb::ERROR_INTERNAL_ERROR;
            }
            inputTensorObjects_.at(listType).push_back(aclnnTensor);
            inputTensorVectors_.at(listType).push_back(aclnnTensor->tensor);
        }
        aclnnVariantPack.aclInTensors.at(atbTensorIdx) = inputTensorObjects_.at(listType).front();
        ++atbTensorIdx;
    }
    aclTensorList *tensorList = aclCreateTensorList(
        inputTensorVectors_.at(listType).data(), inputTensorVectors_.at(listType).size());
    if (tensorList == nullptr) {
        return atb::ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack.aclInTensorList.push_back(tensorList);
    return atb::NO_ERROR;
}

atb::Status MegaMoeOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    if (param_.weight1TensorNum == 0 || param_.weight2TensorNum == 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " MegaMoe weight1 and weight2 TensorList must not be empty");
        return atb::ERROR_INVALID_PARAM;
    }
    if (variantPack.inTensors.size() != GetInputNum()) {
        ATB_SPEED_LOG_ERROR(opName_ << " MegaMoe input number mismatch, expect " << GetInputNum()
                                    << ", got " << variantPack.inTensors.size());
        return atb::ERROR_INVALID_PARAM;
    }

    inputTensorVectors_.clear();
    inputTensorVectors_.resize(kMegaMoeTensorListNum);
    inputTensorObjects_.clear();
    inputTensorObjects_.resize(kMegaMoeTensorListNum);
    inputTensorListIdx_.clear();
    inputTensorListIdx_.resize(kMegaMoeTensorListNum, -1);

    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    aclnnVariantPack.aclInTensorList.clear();

    aclnnVariantPack.aclInTensors.at(kAclContextTensorIdx) = CreateTensor(
        variantPack.inTensors.at(kAclContextTensorIdx), kAclContextTensorIdx);
    aclnnVariantPack.aclInTensors.at(kAclXTensorIdx) = CreateTensor(
        variantPack.inTensors.at(kAclXTensorIdx), kAclXTensorIdx);
    aclnnVariantPack.aclInTensors.at(kAclTopkIdsTensorIdx) = CreateTensor(
        variantPack.inTensors.at(kAclTopkIdsTensorIdx), kAclTopkIdsTensorIdx);
    aclnnVariantPack.aclInTensors.at(kAclTopkWeightsTensorIdx) = CreateTensor(
        variantPack.inTensors.at(kAclTopkWeightsTensorIdx), kAclTopkWeightsTensorIdx);
    for (uint32_t tensorIdx = 0; tensorIdx < kMegaMoeRequiredTensorNum; ++tensorIdx) {
        if (aclnnVariantPack.aclInTensors.at(tensorIdx)->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }

    uint32_t atbTensorIdx = kMegaMoeRequiredTensorNum;
    CHECK_OPERATION_STATUS_RETURN(AppendAclNNInputTensorList(
        variantPack, param_.weight1TensorNum, kWeight1TensorListIdx, atbTensorIdx));
    CHECK_OPERATION_STATUS_RETURN(AppendAclNNInputTensorList(
        variantPack, param_.weight2TensorNum, kWeight2TensorListIdx, atbTensorIdx));
    CHECK_OPERATION_STATUS_RETURN(AppendAclNNInputTensorList(
        variantPack, param_.weightScales1TensorNum, kWeightScales1TensorListIdx, atbTensorIdx));
    CHECK_OPERATION_STATUS_RETURN(AppendAclNNInputTensorList(
        variantPack, param_.weightScales2TensorNum, kWeightScales2TensorListIdx, atbTensorIdx));
    CHECK_OPERATION_STATUS_RETURN(AppendAclNNInputTensorList(
        variantPack, param_.bias1TensorNum, kBias1TensorListIdx, atbTensorIdx));
    CHECK_OPERATION_STATUS_RETURN(AppendAclNNInputTensorList(
        variantPack, param_.bias2TensorNum, kBias2TensorListIdx, atbTensorIdx));

    if (param_.hasXActiveMask) {
        aclnnVariantPack.aclInTensors.at(atbTensorIdx) =
            CreateTensor(variantPack.inTensors.at(atbTensorIdx), kAclXActiveMaskTensorIdx);
        if (aclnnVariantPack.aclInTensors.at(atbTensorIdx)->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

aclTensorList *MegaMoeOperation::GetAclInTensorList(
    const AclNNVariantPack &aclnnVariantPack, uint32_t listType) const
{
    if (listType >= inputTensorListIdx_.size() || inputTensorListIdx_.at(listType) < 0) {
        return nullptr;
    }
    return aclnnVariantPack.aclInTensorList.at(static_cast<uint32_t>(inputTensorListIdx_.at(listType)));
}

int MegaMoeOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    static const auto aclnnMegaMoeGetWorkspaceSize =
        reinterpret_cast<AclnnMegaMoeGetWorkspaceSizeFunc>(GetOpApiFuncAddr("aclnnMegaMoeGetWorkspaceSize"));
    if (aclnnMegaMoeGetWorkspaceSize == nullptr) {
        return kMegaMoeAclnnNotFound;
    }

    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMegaMoeGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(kAclContextTensorIdx)->tensor,
        aclnnVariantPack.aclInTensors.at(kAclXTensorIdx)->tensor,
        aclnnVariantPack.aclInTensors.at(kAclTopkIdsTensorIdx)->tensor,
        aclnnVariantPack.aclInTensors.at(kAclTopkWeightsTensorIdx)->tensor,
        GetAclInTensorList(aclnnVariantPack, kWeight1TensorListIdx),
        GetAclInTensorList(aclnnVariantPack, kWeight2TensorListIdx),
        GetAclInTensorList(aclnnVariantPack, kWeightScales1TensorListIdx),
        GetAclInTensorList(aclnnVariantPack, kWeightScales2TensorListIdx),
        GetAclInTensorList(aclnnVariantPack, kBias1TensorListIdx),
        GetAclInTensorList(aclnnVariantPack, kBias2TensorListIdx),
        param_.hasXActiveMask ? aclnnVariantPack.aclInTensors.at(GetInputNum() - 1)->tensor : nullptr,
        param_.moeExpertNum,
        param_.epWorldSize,
        param_.cclBufferSize,
        param_.maxRecvTokenNum,
        param_.dispatchQuantMode,
        param_.dispatchQuantOutDtype,
        param_.combineQuantMode,
        param_.commAlg.data(),
        param_.numMaxTokensPerRank,
        param_.activation.data(),
        param_.activationClamp,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                                << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                                << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MegaMoeOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMegaMoe start");
    static const auto aclnnMegaMoe = reinterpret_cast<AclnnMegaMoeFunc>(GetOpApiFuncAddr("aclnnMegaMoe"));
    if (aclnnMegaMoe == nullptr) {
        return kMegaMoeAclnnNotFound;
    }

    int ret = aclnnMegaMoe(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMegaMoe end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed
