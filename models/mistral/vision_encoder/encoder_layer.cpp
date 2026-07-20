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

#include "models/mistral/vision_encoder/encoder_layer.h"

#include <map>
#include <string>
#include <vector>

#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace mistral {

// ============================================================
//  Tensor name lists for graph construction
// ============================================================

static std::map<std::string, std::vector<std::string>>
GetMistral3LayerTensorCandidates() {
  std::map<std::string, std::vector<std::string>> candidates = {
      {"default_weight",
       {"in_input_norm_weight", "in_qkv_weight", "in_attn_proj_weight",
        "in_post_norm_weight", "in_mlp_gate_weight",
        "in_mlp_down_weight"}},
      {"default_input",
       {"in_hidden_states", "in_cos_embedding", "in_sin_embedding",
        "in_seq_len"}},
      {"out", {"out"}},
      {"intermediate_out",
       {"input_norm_out", "qkv_linear_out", "intermediate_q",
        "intermediate_k", "intermediate_v", "intermediate_rope_q",
        "intermediate_rope_k", "intermediate_atten_out",
        "intermediate_proj_out", "intermediate_add_out",
        "intermediate_post_norm_out", "intermediate_gateup_out",
        "intermediate_silu_out", "intermediate_down_out"}},
  };
  return candidates;
}

static std::map<std::string, uint32_t> ConstructTensorMap(
    const MistralVisionEncoderLayerParam &param, uint32_t &inTensorNum,
    uint32_t &outTensorNum, uint32_t &internalTensorNum) {
  auto candidates = GetMistral3LayerTensorCandidates();
  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {};

  common::AddTensorToList(candidates, "default_weight", inTensorList);
  common::AddTensorToList(candidates, "default_input", inTensorList);
  common::AddTensorToList(candidates, "out", outTensorList);
  common::AddTensorToList(candidates, "intermediate_out",
                          intermediateTensorList);

  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return common::GetTensorMap(inTensorList, outTensorList,
                              intermediateTensorList);
}

// ============================================================
//  Attention block: Norm → QKV → Split → Rope → Attn → OutProj → Add
// ============================================================

static atb::Status BuildAttentionBlock(
    const MistralVisionEncoderLayerParam &param,
    const std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph) {
  atb::Node inputNormNode;
  atb::Node qkvLinearNode;
  atb::Node splitNode;
  atb::Node ropeNode;
  atb::Node selfAttentionNode;
  atb::Node outProjNode;
  atb::Node selfResidualAddNode;

  // Input Norm (RMSNorm)
  CHECK_OPERATION_STATUS_RETURN(
      CreateNormOperation(param, &inputNormNode.operation));
  inputNormNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", "in_input_norm_weight"});
  inputNormNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"input_norm_out"});
  opGraph.nodes.push_back(inputNormNode);

  // QKV Linear (no bias for Pixtral ViT)
  CHECK_OPERATION_STATUS_RETURN(
      CreateLinearOperation(&qkvLinearNode.operation, false));
  qkvLinearNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"input_norm_out", "in_qkv_weight"});
  qkvLinearNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  opGraph.nodes.push_back(qkvLinearNode);

  // Split QKV
  CHECK_OPERATION_STATUS_RETURN(
      CreateSplitOperation(&splitNode.operation));
  splitNode.inTensorIds =
      common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  splitNode.outTensorIds = common::GetTensorIdxList(
      tensorMap,
      {"intermediate_q", "intermediate_k", "intermediate_v"});
  opGraph.nodes.push_back(splitNode);

  // RoPE
  CHECK_OPERATION_STATUS_RETURN(CreateRopeOperation(&ropeNode.operation));
  ropeNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_q", "intermediate_k", "in_cos_embedding",
                  "in_sin_embedding", "in_seq_len"});
  ropeNode.outTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_rope_q", "intermediate_rope_k"});
  opGraph.nodes.push_back(ropeNode);

  // Self Attention (PA_ENCODER, packed attention with cu_seqlens)
  CHECK_OPERATION_STATUS_RETURN(
      CreateSelfAttentionOperation(param, &selfAttentionNode.operation));
  selfAttentionNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_rope_q", "intermediate_rope_k",
                  "intermediate_v", "in_seq_len"});
  selfAttentionNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"intermediate_atten_out"});
  opGraph.nodes.push_back(selfAttentionNode);

  // Output Projection (no bias for Pixtral ViT)
  CHECK_OPERATION_STATUS_RETURN(
      CreateLinearOperation(&outProjNode.operation, false));
  outProjNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_atten_out", "in_attn_proj_weight"});
  outProjNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"intermediate_proj_out"});
  opGraph.nodes.push_back(outProjNode);

  // Residual Add
  CHECK_OPERATION_STATUS_RETURN(
      CreateAddOperation(&selfResidualAddNode.operation));
  selfResidualAddNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", "intermediate_proj_out"});
  selfResidualAddNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"intermediate_add_out"});
  opGraph.nodes.push_back(selfResidualAddNode);

  return atb::NO_ERROR;
}

// ============================================================
//  MLP block: PostNorm → GateUp → SwiGLU → Down → Add
// ============================================================

static atb::Status BuildMLPBlock(
    const MistralVisionEncoderLayerParam &param,
    const std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph) {
  atb::Node postNormNode;
  atb::Node gateUpNode;
  atb::Node siluNode;
  atb::Node downNode;
  atb::Node mlpResidualAddNode;

  // Post Attention Norm (RMSNorm)
  CHECK_OPERATION_STATUS_RETURN(
      CreateNormOperation(param, &postNormNode.operation));
  postNormNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_add_out", "in_post_norm_weight"});
  postNormNode.outTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_post_norm_out"});
  opGraph.nodes.push_back(postNormNode);

  // Gate+Up Linear (combined weight, no bias)
  CHECK_OPERATION_STATUS_RETURN(
      CreateLinearOperation(&gateUpNode.operation, false));
  gateUpNode.inTensorIds = common::GetTensorIdxList(
      tensorMap,
      {"intermediate_post_norm_out", "in_mlp_gate_weight"});
  gateUpNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"intermediate_gateup_out"});
  opGraph.nodes.push_back(gateUpNode);

  // SwiGLU Activation
  CHECK_OPERATION_STATUS_RETURN(
      CreateActivateOperation(&siluNode.operation,
                              atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD));
  siluNode.inTensorIds =
      common::GetTensorIdxList(tensorMap, {"intermediate_gateup_out"});
  siluNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"intermediate_silu_out"});
  opGraph.nodes.push_back(siluNode);

  // Down Linear (no bias)
  CHECK_OPERATION_STATUS_RETURN(
      CreateLinearOperation(&downNode.operation, false));
  downNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_silu_out", "in_mlp_down_weight"});
  downNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"intermediate_down_out"});
  opGraph.nodes.push_back(downNode);

  // MLP Residual Add
  CHECK_OPERATION_STATUS_RETURN(
      CreateAddOperation(&mlpResidualAddNode.operation));
  mlpResidualAddNode.inTensorIds = common::GetTensorIdxList(
      tensorMap, {"intermediate_down_out", "intermediate_add_out"});
  mlpResidualAddNode.outTensorIds =
      common::GetTensorIdxList(tensorMap, {"out"});
  opGraph.nodes.push_back(mlpResidualAddNode);

  return atb::NO_ERROR;
}

// ============================================================
//  EncoderLayer entry point
// ============================================================

atb::Status EncoderLayer(const MistralVisionEncoderLayerParam &param,
                         atb::Operation **operation) {
  atb::GraphParam opGraph;
  opGraph.name = "Mistral3_Vision_Encoder_Layer";

  // Build tensor map
  std::map<std::string, uint32_t> tensorMap =
      ConstructTensorMap(param, opGraph.inTensorNum, opGraph.outTensorNum,
                         opGraph.internalTensorNum);

  // Build attention and MLP blocks
  CHECK_OPERATION_STATUS_RETURN(
      BuildAttentionBlock(param, tensorMap, opGraph));
  CHECK_OPERATION_STATUS_RETURN(
      BuildMLPBlock(param, tensorMap, opGraph));

  // Shape inference: output shape = input hidden_states shape
  opGraph.inferShapeFunc =
      [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
         atb::SVector<atb::TensorDesc> &outTensorDescs) {
        if (!inTensorDescs.empty() && !outTensorDescs.empty()) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
      };

  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}

}  // namespace mistral
}  // namespace atb_speed
