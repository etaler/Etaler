#pragma once

#include <map>
#include <any>
#include <string>

#include "Etaler_export.h"

namespace et
{

using StateDict = std::map<std::string, std::any>;

void ETALER_EXPORT save(const StateDict& dict, const std::string& path);
StateDict ETALER_EXPORT load(const std::string& path);

}
