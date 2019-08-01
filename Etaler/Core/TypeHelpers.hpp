#pragma once

#include <type_traits>
#include <string>

#include "Etaler_export.h"

//A handy macro to call duplicated const version of the non const function.
#define call_const(func) [&](){\
	auto r = const_cast<const std::remove_reference<decltype(*this)>::type*>(this)->func();\
	using T = decltype(r); \
	if constexpr(std::is_pointer<T>::value) \
		return const_cast<typename std::remove_const<typename std::remove_pointer<T>::type>::type*>(r);\
	else \
		return r;\
}()



namespace et
{

std::string ETALER_EXPORT demangle(const char* name);

template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

template<template<typename...> class Ref, typename... Args>
const bool is_specialization_v = is_specialization<Ref<Args...>, Ref>::value;

}