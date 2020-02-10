#include <memory>

#include <Etaler/Backends/CPUBackend.hpp>
#include "DefaultBackend.hpp"

namespace et
{
std::shared_ptr<Backend> g_default_backend_hold;
Backend* g_default_backend = nullptr;
}

using namespace et;

Backend* et::defaultBackend()
{
	using DefaultBackendType = CPUBackend;
	if(g_default_backend == nullptr) {
		if((bool)g_default_backend_hold == false)
			g_default_backend_hold = std::make_shared<DefaultBackendType>();

		g_default_backend = g_default_backend_hold.get();
	}
	return g_default_backend;
}
