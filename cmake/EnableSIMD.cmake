if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
	message(STATUS "Building For x86/x64")
	if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
		message(STATUS "Compiler is GCC/Clang: Enabling SIMD via SSE and AVX")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -mavx")
	else() # MSVC
	message(STATUS "Compiler is MSVC: Enabling SIMD via AVX")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
	endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	message(STATUS "Building for Aarch64: SIMD enabled by compiler by default")
	# GCC and clang enables NEON/Advanced SIMD by default
else()
	#TODO: Add support for PPCLE, RISC-V, etc... support
	message(STATUS "Building for ${CMAKE_SYSTEM_PROCESSOR}: SIMD might not be enabled by EnableSIMD.cmake")
endif ()
