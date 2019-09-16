#include "TypeHelpers.hpp"

using namespace et;

#ifdef HAVE_CXA_DEMANGLE //Set by CMake if avaliable

#include <cxxabi.h>
#include <stdlib.h>

std::string et::demangle(const char* name) {

    int status = -4;

    char* res = abi::__cxa_demangle(name, NULL, NULL, &status);
    const char* const demangled_name = (status==0)?res:name;
    std::string ret_val(demangled_name);

    free(res);

    return ret_val;
}

#else

#if _MSC_VER
#pragma message ("warning: CXA_DEMANGLE API not avaliable. Type demangling is not enabled. (Worse exception messages)")
#else
#warning CXA_DEMANGLE API not avaliable. Type demangling is not enabled. (Worse exception messages)
#endif

std::string et::demangle(const char* name) {
	return std::string(name);
}


#endif