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

#pragma once

#include <string>
#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace mistral {

struct MistralVisionEncoderLayerParam {
  bool isBF16 = false;
  bool supportLcoc = false;
  bool supportLora = false;
  bool loraEnableGMM = false;
  std::string backend = "hccl";
  int rank = 0;
  int worldSize = 1;
  int quantType = 0;
  int quantGroupSize = 64;
  int numAttentionHeadsPerRank = 0;
  int hiddenSizePerAttentionHead = 0;
  int numKeyValueHeadsPerRank = 0;
  float rmsNormEps = 1e-5f;

  std::vector<int> seqLen;
  std::vector<int> tokenOffset;
  std::vector<int> packQuantType = {};
  std::vector<int> linearQuantType = {};
  std::vector<int> linearTransposeType;
};

atb::Status CreateNormOperation(
    const MistralVisionEncoderLayerParam &param, atb::Operation **op);

atb::Status CreateLinearOperation(atb::Operation **op, bool hasBias = false,
                                  bool istransB = true);

atb::Status CreateSplitOperation(atb::Operation **op);

atb::Status CreateRopeOperation(atb::Operation **op);

atb::Status CreateSelfAttentionOperation(
    const MistralVisionEncoderLayerParam &param, atb::Operation **op);

atb::Status CreateActivateOperation(atb::Operation **op,
                                    atb::infer::ActivationType type);

atb::Status CreateAddOperation(atb::Operation **op);

}  // namespace mistral
}  // namespace atb_speed
