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

}

// Asserts
#define et_assert_with_message(expression, msg) do{if((expression) == false) {std::cerr << msg; std::abort();}}while(0)
#define et_assert_no_message(expression) do{if((expression) == false){std::cerr << "Assertion " << #expression << " failed"; std::abort();}}while(0)
#define __GetEtAssrtyMacro(_1,_2,NAME,...) NAME
//et_assert is basically C assert but not efficeted by the NDEBUG flag. Use assert for debug, et_assert for possible user screw-ups.
#define et_assert(...) __GetEtAssrtyMacro(__VA_ARGS__ ,et_assert_with_message, et_assert_no_message, nullptr)(__VA_ARGS__)

// Checks. These are higer level APIs. Which when condition fails, they unwind the stack and throws an excpetion
#define et_check_with_message(expression, msg) do{if((expression) == false) {throw EtError(msg);}}while(0)
#define et_check_no_message(expression) do{if((expression) == false){throw EtError(std::string("Check ")+#expression+" failed");}}while(0)
#define __GetEtCheckyMacro(_1,_2,NAME,...) NAME
#define et_check(...) __GetEtCheckyMacro(__VA_ARGS__ ,et_check_with_message, et_check_no_message, nullptr)(__VA_ARGS__)


//Replaces stupid C asserts
#ifdef NDEBUG
	#define ASSERT(x) do { (void)sizeof(x);} while (0)
#else
	#include <assert.h>
	#define ASSERT(x) assert(x)
#endif
