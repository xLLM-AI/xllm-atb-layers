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

#include "operations/aclnn/utils/op_api_resolver.h"

#include <dlfcn.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
namespace {

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
            ATB_SPEED_LOG_DEBUG("dlopen " << libPath << " failed, error:" << dlerror());
            continue;
        }
        libs.push_back({libPath, handle});
    }
    return libs;
}

} // namespace

void *GetOpApiFuncAddr(const char *apiName)
{
    static const std::vector<OpApiLib> libs = OpenOpApiLibs();
    for (const OpApiLib &lib : libs) {
        dlerror();
        void *funcAddr = dlsym(lib.handle, apiName);
        const char *error = dlerror();
        if (error == nullptr && funcAddr != nullptr) {
            ATB_SPEED_LOG_DEBUG(apiName << " is found in " << lib.path);
            return funcAddr;
        }
    }
    ATB_SPEED_LOG_ERROR("can not find " << apiName << " in opapi libs");
    return nullptr;
}

} // namespace common
} // namespace atb_speed
