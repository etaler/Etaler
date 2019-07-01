#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>


#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Core/DefaultBackend.hpp>

int main( int argc, char* argv[] )
{
	Catch::Session session;
	size_t platform_id = 0;
	size_t device_id = 0;

	using namespace Catch::clara;
	auto cli  = session.cli() 
		| Opt(platform_id, "platform ID" )
		["-p"]["--platform"]
		("Which OpenCL platform to use")
		
		| Opt(device_id, "device ID")
		["--device"]
		("Which OpenCL device to use");
	session.cli(cli); 
	int return_code = session.applyCommandLine(argc, argv);
	if(return_code != 0) // Indicates a command line error
		return return_code;
	
	auto backend = std::make_shared<et::OpenCLBackend>(platform_id, device_id);
	et::setDefaultBackend(backend.get());

	std::cout << "Running with backend: " 	<< backend->name() << std::endl;
	// std::cout << "\n=========== OpenCL infomation ===========\n"
	// 	<< backend->deviceInfo() << std::endl;

	return session.run();
}
