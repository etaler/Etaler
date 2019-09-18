#pragma once

#include <exception>
#include <string>
#include <iostream>
#include <memory>
#include <type_traits>

#include "TypeHelpers.hpp"

//Use EtError for recoverable exceptions. Avoid using EtError in core functionalitys
namespace et
{
class EtError : public std::exception
{
public:
	explicit EtError(const std::string &msg) : msg_(msg) {}
	const char *what() const throw() override { return msg_.c_str(); }

private:
	std::string msg_;
};

//No namespace
#define et_assert_with_message(expression, msg) do{if((expression) == false) {std::cerr << msg; std::abort();}}while(0)
#define et_assert_no_message(expression) do{if((expression) == false){std::cerr << "Assertion " << #expression << " failed"; std::abort();}}while(0)
#define __GetAtAssrtyMacro(_1,_2,NAME,...) NAME

//et_assert is basically C assert but not efficeted by the NDEBUG flag. Use assert for debug, et_assert for possible user screw-ups.
#define et_assert(...) __GetAtAssrtyMacro(__VA_ARGS__ ,et_assert_with_message, et_assert_no_message)(__VA_ARGS__)

template<typename T2, typename T1>
bool points_to(const T1 p)
{
	if constexpr (std::is_pointer<T1>::value)
		return dynamic_cast<const T2*>(p) != nullptr;
	//TODO: Replace this with C++20's concepts. Check if p.get() is returning a pointer
	else if(is_specialization_v<T1, std::shared_ptr> || is_specialization_v<T1, std::unique_ptr>
		|| is_specialization_v<T1, std::weak_ptr>)
		return dynamic_cast<const T2*>(p.get()) != nullptr;
	else
		return false;
}
}

//Replaces stupid C asserts
#ifdef NDEBUG
	#define ASSERT(x) do { (void)sizeof(x);} while (0)
#else
	#include <assert.h>
	#define ASSERT(x) assert(x)
#endif
