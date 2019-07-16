#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>


//#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Core/DefaultBackend.hpp>

int main( int argc, char* argv[] )
{
	std::cout << "Running with backend: " << et::defaultBackend()->name() << std::endl;

	int result = Catch::Session().run( argc, argv );

	return result;
}
