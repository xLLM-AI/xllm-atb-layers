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

#ifndef ATB_SPEED_PLUGIN_OP_API_RESOLVER_H
#define ATB_SPEED_PLUGIN_OP_API_RESOLVER_H

namespace atb_speed {
namespace common {

/// Error code returned by aclnn adapters when an op-api entry cannot be
/// resolved from any op-api library.
constexpr int kAclnnOpApiNotFound = -1;

/// Resolve an aclnn op-api entry symbol from the op-api libraries with the
/// same search order as the torch_npu op loader: each vendor in
/// ASCEND_CUSTOM_OPP_PATH (in order), then each vendor in
/// $ASCEND_OPP_PATH/vendors per config.ini load_priority, then the
/// libcust_opapi.so / libopapi.so sonames via the default linker path.
///
/// Ops whose definition keeps evolving in an ops-transformer vendor package
/// (e.g. aclnnSparseFlashAttention, aclnnMegaMoe) must resolve their entry
/// through this helper instead of calling the CANN built-in entry directly:
/// the built-in wrapper marshals parameters per the CANN release's older op
/// definition, which diverges from the vendor opmaster and kernel selected by
/// the aclInit registry and produces silently wrong results.
///
/// \param apiName The aclnn symbol name, e.g. "aclnnMegaMoe".
/// \return The function address, or nullptr when not found in any library.
void *GetOpApiFuncAddr(const char *apiName);

/// Typed variant of GetOpApiFuncAddr.
///
/// \param apiName The aclnn symbol name.
/// \return The typed function pointer, or nullptr when not found.
template <typename Func>
Func GetOpApiFunc(const char *apiName)
{
    return reinterpret_cast<Func>(GetOpApiFuncAddr(apiName));
}

} // namespace common
} // namespace atb_speed
#endif
