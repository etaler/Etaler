#include <type_traits>

namespace et
{
//Using STL coding style in this file
struct null_t {};

template <typename T, typename U>
struct type_list_node
{
    using head = T;
    using tail = U;
};

template <typename ... Ts> struct type_list;

// Case: Normal recursion. Consume one type per call.
template <typename T, typename ... REST>
struct type_list<T, REST...> { 
    using type = type_list_node<T, typename type_list<REST...>::type>;
};

// Case: Recursion abort, because the list of types ran empty
template <>
struct type_list<> { using type = null_t; };

template <typename ... Ts>
using type_list_t = typename type_list<Ts...>::type;

}