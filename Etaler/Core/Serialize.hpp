#pragma once

#include <map>
#include <any>
#include <string>

namespace et
{

using StateDict = std::map<std::string, std::any>;

void save(const StateDict& dict, const std::string& path);
StateDict load(const std::string& path);

}
