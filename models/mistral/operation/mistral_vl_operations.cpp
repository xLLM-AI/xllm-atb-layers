/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "models/mistral/operation/mistral_vl_operations.h"

#include <cmath>

namespace atb_speed {
namespace mistral {

atb::Status CreateNormOperation(
    const MistralVisionEncoderLayerParam &param, atb::Operation **op) {
  atb::infer::RmsNormParam normParam;
  normParam.layerType =
      atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  normParam.normParam.epsilon = param.rmsNormEps;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(normParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateLinearOperation(atb::Operation **op, bool hasBias,
                                  bool istransB) {
  atb::infer::LinearParam linearParam;
  linearParam.transposeA = false;
  linearParam.transposeB = istransB;
  linearParam.hasBias = hasBias;
  linearParam.outDataType = aclDataType::ACL_DT_UNDEFINED;
  linearParam.enAccum = false;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateSplitOperation(atb::Operation **op) {
  atb::infer::SplitParam splitParam;
  splitParam.splitDim = -1;
  splitParam.splitNum = 3;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateRopeOperation(atb::Operation **op) {
  atb::infer::RopeParam ropeParam;
  ropeParam.rotaryCoeff = 2;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(ropeParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateSelfAttentionOperation(
    const MistralVisionEncoderLayerParam &param, atb::Operation **op) {
  atb::infer::SelfAttentionParam attentionParam;
  attentionParam.headNum = param.numAttentionHeadsPerRank;
  // Mistral3 Pixtral ViT: full attention, kv heads = q heads
  attentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
  attentionParam.calcType =
      atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
  attentionParam.maskType =
      atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED;
  attentionParam.qkScale =
      1.0 / sqrt(static_cast<double>(param.hiddenSizePerAttentionHead));
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(attentionParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateActivateOperation(atb::Operation **op,
                                    atb::infer::ActivationType type) {
  atb::infer::ActivationParam actParam;
  actParam.activationType = type;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(actParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateAddOperation(atb::Operation **op) {
  atb::infer::ElewiseParam addParam;
  addParam.elewiseType =
      atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, op));
  return atb::NO_ERROR;
}

}  // namespace mistral
}  // namespace atb_speed
