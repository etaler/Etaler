#include <catch2/catch_session.hpp>


//#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Core/DefaultBackend.hpp>

int main( int argc, char* argv[] )
{
	et::enableTraceOnException(false); // Cleaner exception message
	std::cout << "Running with backend: " << et::defaultBackend()->name() << std::endl;

	int result = Catch::Session().run( argc, argv );

	return result;
}
