#pragma once

#include <memory>
#include <string>

#include "Shape.hpp"
#include "DType.hpp"
#include "Error.hpp"

using std::size_t;

namespace et
{

struct TensorImpl;
struct Backend;

struct Backend : public std::enable_shared_from_this<Backend>
{
	virtual ~Backend() = default;
	virtual std::shared_ptr<TensorImpl> createTensor(const Shape& shape, DType dtype, const void* data = nullptr) {throw notImplemented("createTensor");};
	virtual void releaseTensor(TensorImpl* pimpl) {throw notImplemented("releaseTensor");};
	virtual void sync() const {} //Default empty implemention. For async backends
	virtual void overlapScore(const TensorImpl* x, const TensorImpl* connections,
		const TensorImpl* permeances, float connected_permeance, size_t active_threshold, TensorImpl* y, bool has_unconnected_synapse=true) {throw notImplemented("overlapScore");}
	virtual void learnCorrilation(const TensorImpl* x, const TensorImpl* learn,
		const TensorImpl* connections, TensorImpl* permeances, float perm_inc, float perm_dec) {throw notImplemented("learnCorrilation");}
	virtual void globalInhibition(const TensorImpl* x, TensorImpl* y, float fraction) {throw notImplemented("globalInhibition");}
	virtual std::shared_ptr<TensorImpl> cast(const TensorImpl* x, DType toType) {throw notImplemented("cast");}
	virtual void copyToHost(const TensorImpl* pimpl, void* dest) {throw notImplemented("copyToHost");}
	virtual std::string name() const {return "BaseBackend";}
	virtual std::shared_ptr<TensorImpl> copy(const TensorImpl* x) {throw notImplemented("copy");}
	virtual void sortSynapse(TensorImpl* connections, TensorImpl* permeances) {throw notImplemented("sortSynapse");}
	virtual void applyBurst(const TensorImpl* x, TensorImpl* y) {throw notImplemented("applyBurst");}
	virtual void reverseBurst(TensorImpl* x) {throw notImplemented("reverseBurst");}
	virtual void growSynapses(const TensorImpl* x, TensorImpl* y, TensorImpl* connections
		, TensorImpl* permeances, float initial_perm) {throw notImplemented("growSynapses");}

	inline EtError notImplemented(std::string func) const { return EtError(func + " not implemented on backend: " + name()); }


};

}
