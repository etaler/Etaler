#include <memory>

#include <Etaler/Backends/CPUBackend.hpp>

namespace et
{
std::shared_ptr<Backend> g_fallback_backend;
Backend* g_default_backend;
}

using namespace et;

Backend* et::defaultBackend()
{
	if(g_default_backend == nullptr) {
		//std::cerr << "Error: defaultBackend() called before setting the default backend.\n";
		//abort();
		if((bool)g_fallback_backend == false)
			g_fallback_backend = std::make_shared<et::CPUBackend>();

		g_default_backend = g_fallback_backend.get();
	}
	return g_default_backend;
}
