#pragma once

#include <vector>
#include <string>

namespace et
{

inline std::vector<std::string> split(const std::string& str, char delim = ',')
{
	std::size_t current, previous = 0;
	std::vector<std::string> cont;
	current = str.find(delim);
	while (current != std::string::npos) {
		cont.push_back(str.substr(previous, current - previous));
		previous = current + 1;
		current = str.find(delim, previous);
	}
	cont.push_back(str.substr(previous, current - previous));
	return cont;
}

}