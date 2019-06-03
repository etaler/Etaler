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

	virtual void sync() const {} //Default empty implemention. For async backends
	virtual std::shared_ptr<TensorImpl> cellActivity(const TensorImpl* x, const TensorImpl* connections,
		const TensorImpl* permeances, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse=true) {throw notImplemented("overlapScore");}
	virtual void learnCorrilation(const TensorImpl* x, const TensorImpl* learn,
		const TensorImpl* connections, TensorImpl* permeances, float perm_inc, float perm_dec
		, bool has_unconnected_synapse=true) {throw notImplemented("learnCorrilation");}
	virtual std::shared_ptr<TensorImpl> globalInhibition(const TensorImpl* x, float fraction) {throw notImplemented("globalInhibition");}
	virtual std::shared_ptr<TensorImpl> cast(const TensorImpl* x, DType toType) {throw notImplemented("cast");}
	virtual void copyToHost(const TensorImpl* pimpl, void* dest) {throw notImplemented("copyToHost");}
	virtual std::string name() const {return "BaseBackend";}
	virtual std::shared_ptr<TensorImpl> copy(const TensorImpl* x) {throw notImplemented("copy");}
	virtual void sortSynapse(TensorImpl* connections, TensorImpl* permeances) {throw notImplemented("sortSynapse");}
	virtual std::shared_ptr<TensorImpl> burst(const TensorImpl* x, const TensorImpl* s) {throw notImplemented("burst");}
	virtual std::shared_ptr<TensorImpl> reverseBurst(const TensorImpl* x) {throw notImplemented("reverseBurst");}
	virtual void growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
		, TensorImpl* permeances, float initial_perm) {throw notImplemented("growSynapses");}
	virtual void decaySynapses(TensorImpl* connections, TensorImpl* permeances, float threshold) {throw notImplemented("decaySynapses");}
	virtual std::shared_ptr<TensorImpl> from(const TensorImpl* x) {throw notImplemented("from");}

	virtual std::shared_ptr<TensorImpl> realize(const TensorImpl* x) {throw notImplemented("realize");}
	virtual void assign(TensorImpl* dest, const TensorImpl* src) {throw notImplemented("assign");}
	virtual std::shared_ptr<TensorImpl> sum(const TensorImpl* x, size_t chunk_size, DType dtype=DType::Unknown) { throw notImplemented("sum");}

	//Unary operations
	virtual std::shared_ptr<TensorImpl> exp(const TensorImpl* x) { throw notImplemented("exp");}
	virtual std::shared_ptr<TensorImpl> negate(const TensorImpl* x) { throw notImplemented("negate");}
	virtual std::shared_ptr<TensorImpl> inverse(const TensorImpl* x) { throw notImplemented("inverse");}
	virtual std::shared_ptr<TensorImpl> log(const TensorImpl* x) { throw notImplemented("log");}
	virtual std::shared_ptr<TensorImpl> logical_not(const TensorImpl* x) { throw notImplemented("logical_not");}

	//Binary operations
	virtual std::shared_ptr<TensorImpl> add(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("add");}
	virtual std::shared_ptr<TensorImpl> subtract(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("subtract");}
	virtual std::shared_ptr<TensorImpl> mul(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("mul");}
	virtual std::shared_ptr<TensorImpl> div(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("div");}
	virtual std::shared_ptr<TensorImpl> equal(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("equal");}
	virtual std::shared_ptr<TensorImpl> greater(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("greater");}
	virtual std::shared_ptr<TensorImpl> lesser(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("lesster");}
	virtual std::shared_ptr<TensorImpl> logical_and(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("and");}
	virtual std::shared_ptr<TensorImpl> logical_or(const TensorImpl* x1, const TensorImpl* x2) { throw notImplemented("or");}

	inline EtError notImplemented(std::string func) const { return EtError(func + " not implemented on backend: " + name()); }
};

}

