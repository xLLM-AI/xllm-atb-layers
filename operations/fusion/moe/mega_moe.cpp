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
#include "operations/fusion/moe/mega_moe.h"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>

#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {
namespace {

constexpr uint32_t kMegaMoeOutputNum = 2;
constexpr uint32_t kMegaMoeNodeCount = 1;
constexpr uint32_t kMegaMoeXInputIdx = 1;

std::string GetMegaMoeOperationName(const MegaMoeParam &param)
{
    std::ostringstream opName;
    opName << "MegaMoeOperation"
           << "_moeExpertNum_" << param.moeExpertNum
           << "_epWorldSize_" << param.epWorldSize
           << "_cclBufferSize_" << param.cclBufferSize
           << "_maxRecvTokenNum_" << param.maxRecvTokenNum
           << "_dispatchQuantMode_" << param.dispatchQuantMode
           << "_dispatchQuantOutDtype_" << param.dispatchQuantOutDtype
           << "_combineQuantMode_" << param.combineQuantMode
           << "_commAlg_" << param.commAlg
           << "_numMaxTokensPerRank_" << param.numMaxTokensPerRank
           << "_activation_" << param.activation
           << "_activationClamp_" << std::setprecision(std::numeric_limits<float>::max_digits10)
           << param.activationClamp
           << "_weight1TensorNum_" << param.weight1TensorNum
           << "_weight2TensorNum_" << param.weight2TensorNum
           << "_weightScales1TensorNum_" << param.weightScales1TensorNum
           << "_weightScales2TensorNum_" << param.weightScales2TensorNum
           << "_bias1TensorNum_" << param.bias1TensorNum
           << "_bias2TensorNum_" << param.bias2TensorNum
           << "_hasXActiveMask_" << param.hasXActiveMask;
    return opName.str();
}

atb::Status InferMegaMoeShape(const MegaMoeParam &param,
                              const atb::SVector<atb::TensorDesc> &inTensorDescs,
                              atb::SVector<atb::TensorDesc> &outTensorDescs)
{
    if (param.epWorldSize <= 0) {
        ATB_SPEED_LOG_ERROR("MegaMoe epWorldSize must be greater than 0, got " << param.epWorldSize);
        return atb::ERROR_INVALID_PARAM;
    }

    outTensorDescs.at(DIM0) = inTensorDescs.at(kMegaMoeXInputIdx);
    outTensorDescs.at(DIM1).format = ACL_FORMAT_ND;
    outTensorDescs.at(DIM1).dtype = ACL_INT32;
    outTensorDescs.at(DIM1).shape.dimNum = 1;
    outTensorDescs.at(DIM1).shape.dims[DIM0] = param.moeExpertNum / param.epWorldSize;
    return atb::NO_ERROR;
}

}  // namespace

atb::Status CreateMegaMoeOperation(const MegaMoeParam &param, atb::Operation **operation)
{
    if (param.weight1TensorNum == 0 || param.weight2TensorNum == 0) {
        ATB_SPEED_LOG_ERROR("MegaMoe weight1 and weight2 TensorList must not be empty");
        return atb::ERROR_INVALID_PARAM;
    }
    if (param.epWorldSize <= 0) {
        ATB_SPEED_LOG_ERROR("MegaMoe epWorldSize must be greater than 0, got " << param.epWorldSize);
        return atb::ERROR_INVALID_PARAM;
    }

    atb::GraphParam opGraph;
    opGraph.name = "MegaMoe";
    opGraph.inTensorNum = GetMegaMoeInputNum(param);
    opGraph.outTensorNum = kMegaMoeOutputNum;
    opGraph.internalTensorNum = 0;
    opGraph.nodes.resize(kMegaMoeNodeCount);

    atb::Node &megaMoeNode = opGraph.nodes.at(DIM0);
    megaMoeNode.operation = new MegaMoeOperation(GetMegaMoeOperationName(param), param);
    for (uint32_t tensorIdx = 0; tensorIdx < opGraph.inTensorNum; ++tensorIdx) {
        megaMoeNode.inTensorIds.push_back(tensorIdx);
    }

    megaMoeNode.outTensorIds = {opGraph.inTensorNum, opGraph.inTensorNum + 1};
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        return InferMegaMoeShape(param, inTensorDescs, outTensorDescs);
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

}  // namespace common
}  // namespace atb_speed
