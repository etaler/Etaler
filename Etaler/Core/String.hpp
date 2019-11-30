#pragma once

#include <vector>
#include <string>
#include <sstream>

#include "Shape.hpp"

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

inline std::string hash_string(const std::string& str)
{
	auto hash = std::hash<std::string>()(str);
	std::stringstream ss;
	ss << std::hex << hash;
	return ss.str();
}

inline void replaceAll(std::string& str, const std::string& from, const std::string& to) {
	if(from.empty())
		return;
	size_t start_pos = 0;
	while((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}

class Tensor;
enum class DType;
// A lazy and not bullet-proof hashing function
template <typename ... Args>
inline std::string hashify(const Args& ... args)
{
	auto to_str = [](const auto& v) -> std::string {
		using ValueType = std::decay_t<decltype(v)>;
		// HACK: You should NEVER put a tensor in hashify
		static_assert(std::is_same_v<ValueType, Tensor> == false);


		using namespace std;
		if constexpr(std::is_same_v<ValueType, std::string>)
			return v;
		else if constexpr(std::is_same_v<ValueType, Shape> || std::is_same_v<ValueType, DType>)
			return to_string(v);
		else
			return std::to_string(v);
	};
	std::string concated = ((to_str(args) + " ") + ...);
	return hash_string(concated);
}

}
