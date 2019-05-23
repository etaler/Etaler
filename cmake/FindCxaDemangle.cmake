include(CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES("
#include <cxxabi.h>
int main(void){
    int foo = 0;
    const char *bar = typeid(foo).name();
    int res;
    char *demangled = abi::__cxa_demangle(bar, 0, 0, &res);
}" HAVE_CXA_DEMANGLE)

mark_as_advanced(HAVE_CXA_DEMANGLE)