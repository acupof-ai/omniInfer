// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <string>

namespace umminfer::client {

int run_cli(const std::string& command, int argc, char** argv, std::atomic<bool>* cancel);

}  // namespace umminfer::client
