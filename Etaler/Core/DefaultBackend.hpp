#pragma once

#include "Etaler/Core/Backend.hpp"

#include <memory>

namespace et
{

extern ETALER_EXPORT Backend* g_default_backend;
extern ETALER_EXPORT std::shared_ptr<Backend> g_default_backend_hold;

inline void setDefaultBackend(Backend* backend) {g_default_backend = backend;}
inline void setDefaultBackend(std::shared_ptr<Backend> backend) {g_default_backend_hold = backend; g_default_backend = backend.get();}
ETALER_EXPORT Backend* defaultBackend();

}
