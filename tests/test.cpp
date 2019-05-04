#include "catch2/catch.hpp"

#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Etaler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
#include <Etaler/Encoders/Category.hpp>
#include <Etaler/Encoders/GridCell1d.hpp>
#include <Etaler/Core/Serialize.hpp>

TEST_CASE("Testing Shape", "[Shape]")
{
	using namespace et;
	Shape s = {3,5};

	REQUIRE(s.size() == 2);
	REQUIRE(s.volume() == 15);

	Shape n = s + 7;

	CHECK(n.size() == 3);
	CHECK(n.volume() == 3*5*7);
	CHECK(n == Shape({3,5,7}));
	CHECK(n.contains(5) == true);
	CHECK(n.contains(6) == false);
	CHECK(Shape(n) == n);
	CHECK(s != n);
}

TEST_CASE("Testing Tensor", "[Tensor]")
{
	using namespace et;

	SECTION("Tensor creation") {
		int a[] = {1,2,3,4};
		Tensor t = Tensor({4}, a);
		CHECK(t.dtype() == DType::Int32);
		CHECK(t.shape() == Shape({4}));

		float b[] = {1,2,3,4};
		Tensor q = Tensor({4}, b);
		CHECK(q.dtype() == DType::Float);
		CHECK(q.shape() == Shape({4}));

		bool c[] = {1,1,0,1};
		Tensor r = Tensor({4}, c);
		CHECK(r.dtype() == DType::Bool);
		CHECK(r.shape() == Shape({4}));
	}

	SECTION("Tesnor basic") {
		Tensor t = Tensor({1,2,5,6,7}, DType::Float);

		CHECK(t.size() == 1*2*5*6*7);
		CHECK(t.dtype() == DType::Float);
		CHECK(t.dimentions() == 5);
		CHECK(t.backend() == defaultBackend());
		CHECK(t.dimentions() == t.shape().size());

		//Should not be able to convert from shape {1,2,5,3,7} to {60}
		CHECK_THROWS_AS(t.resize({60}), EtError);

		CHECK_NOTHROW(t.resize({2,5,42}));
	}

	SECTION("Comparsion") {
		int data[] = {1,2,3,4,5};
		Tensor t = Tensor({5}, data);
		Tensor q = Tensor({5}, data);

		CHECK(t.isSame(q));

		Tensor r = Tensor({1, 5}, data);
		CHECK_FALSE(t.isSame(r));
	}

	SECTION("Data Transfer") {
		int data[] = {1,2,3,2,1};
		Tensor t = Tensor({5}, data);
		auto data2 = t.toHost<int32_t>();
		Tensor q = Tensor({5}, data2.data());
		CHECK(t.isSame(q));

		Tensor r = t.copy();
		CHECK(t.isSame(r));
		CHECK(q.isSame(r));
	}

	SECTION("Views") {
		std::vector<int> data(16);
		for(size_t i=0;i<data.size();i++)
			data[i] = i;
		Tensor t = Tensor({4,4}, data.data());

		SECTION("Reshape") {
			CHECK_THROWS(t.reshape({4}));

			CHECK_NOTHROW(t.reshape({16}));
			Tensor q;
			CHECK_NOTHROW(q = t.reshape({4, 4}));

			CHECK(realize(q).isSame(t));
		}

		SECTION("flatten") {
			Tensor q = t.flatten();
			CHECK(q.size() == t.size());
			CHECK(q.dtype() == t.dtype());
			Tensor r = q.reshape({4,4});
			CHECK(realize(r).isSame(t));
		}

		SECTION("Basic indexing/view") {
			CHECK_THROWS(t.view({0,0,0,0,0}));
			CHECK_THROWS(t.view({300}));
			CHECK_THROWS(t.view({0, 300}));
			CHECK_THROWS(t.view({range(100)}));

			Tensor q = t.view({2,2});
			CHECK(q.size() == 1);
			CHECK(q.dimentions() == 1);
			CHECK(realize(q).toHost<int32_t>()[0] == 10);

			Tensor r = t.view({range(2), range(2)});
			CHECK(r.size() == 4);
			CHECK(r.dimentions() == 2);
			int a[] = {0,1,4,5};
			Tensor pred = Tensor({2,2}, a);
			CHECK(realize(r).isSame(pred));
		}

		SECTION("View write back") {
			Tensor q = t.view({range(2),range(2)});
			CHECK_THROWS(q.assign(ones({5,5})));
			Tensor r = ones({2,2});
			CHECK_NOTHROW(q.assign(r));

			int a[] = {1,1,2,3
				,1,1,6,7
				,8,9,10,11
				,12,13,14,15};
			Tensor pred = Tensor({4,4}, a);
			CHECK(t.isSame(pred));
		}
	}
}

TEST_CASE("Testing Encoders", "[Encoder]")
{
	using namespace et;

	SECTION("Scalar Encoder") {
		size_t total_bits = 32;
		size_t num_on_bits = 4;
		Tensor t = encoder::scalar(0.1, 0, 1, total_bits, num_on_bits);
		CHECK(t.size() == 32);
		REQUIRE(t.dtype() == DType::Bool);
		auto v = t.toHost<uint8_t>();
		CHECK(std::accumulate(v.begin(), v.end(), 0) == num_on_bits);
	}

	SECTION("Category Encoder") {
		size_t num_categories = 4;
		size_t bits_per_category = 2;
		Tensor t = encoder::category(0, num_categories, bits_per_category);

		CHECK(t.size() == num_categories*bits_per_category);
		REQUIRE(t.dtype() == DType::Bool);
		auto v = t.toHost<uint8_t>();
		CHECK(std::accumulate(v.begin(), v.end(), 0) == bits_per_category);

		Tensor q = encoder::category(1, num_categories, bits_per_category);
		auto u = q.toHost<uint8_t>();

		size_t overlap = 0;
		for(size_t i=0;i<t.size();i++)
			overlap += v[i] && u[i];
		CHECK(overlap == 0);
	}

	SECTION("GridCell1d Encoder") {
		Tensor t = encoder::gridCell1d(0.1, 16, 1, 16);
		CHECK(t.size() == 16*16);
		auto v = t.toHost<uint8_t>();
		CHECK(std::accumulate(v.begin(), v.end(), 0) == 16);

		Tensor q = encoder::gridCell1d(0.1, 16, 2, 16);
		CHECK(q.size() == 16*16);
		auto u = q.toHost<uint8_t>();
		CHECK(std::accumulate(u.begin(), u.end(), 0) == 32);
	}
}

TEST_CASE("Backend functions", "[Backend]")
{
	using namespace et;

	Backend* b = defaultBackend();

	SECTION("Overlap Score") {
		int32_t synapses[4] = {0, 1, 1, -1};
		Tensor s = Tensor({2,2}, synapses);

		float perm[4] = {0.5, 0.4, 0.7, 0.0};
		Tensor p = Tensor({2,2}, perm);

		uint8_t in[2] = {1,1};
		Tensor x = Tensor({2}, in);

		Tensor y = b->overlapScore(x, s, p, 0.1, 1);
		CHECK(y.size() == 2);

		auto res = y.toHost<int32_t>();
		REQUIRE(res.size() == 2);
		CHECK(res[0] == 2);
		CHECK(res[1] == 1);
	}

	SECTION("Global Inhibition") {
		int32_t in[8] = {0,0,1,2,7,6,5,3};
		Tensor t = Tensor({8}, in);

		Tensor y = b->globalInhibition(t, 0.5);
		CHECK(y.size() == 8);
		uint8_t pred[8] = {0,0,0,0,1,1,1,1};
		Tensor should_be = Tensor({8}, pred);
		CHECK(y.dtype() == DType::Bool);
		CHECK(y.isSame(should_be));
	}

	SECTION("Sort synapse") {
		int a[] = {0,1,2,3, 1,3,2,0, -1,1,2,3, 2,3,1,-1};
		Tensor t = Tensor({4,4}, a);

		float b[] = {0,1,2,3, 1,3,2,0, -1,1,2,3, 2,3,1,-1};
		Tensor q = Tensor({4,4}, b);

		defaultBackend()->sortSynapse(t, q);

		int pred[] = {0,1,2,3, 0,1,2,3, 1,2,3,-1, 1,2,3,-1};
		float pred2[] = {0,1,2,3, 0,1,2,3, 1,2,3,-1, 1,2,3,-1};
		Tensor should_be = Tensor({4,4}, pred);
		Tensor perm_should_be = Tensor({4,4}, pred2);

		CHECK(t.isSame(should_be));
		CHECK(q.isSame(perm_should_be));
	}

	SECTION("Sync") {
		CHECK_NOTHROW(defaultBackend()->sync());
	}

	SECTION("Burst") {
		uint8_t s[] = {0, 0, 0, 0,
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 1, 1, 0};
		Tensor state = Tensor({4,4}, s);

		uint8_t in[] = {1,0,1,1};
		Tensor x = Tensor({4}, in);
		Tensor y = defaultBackend()->applyBurst(x, state);

		uint8_t p[] = {1, 1, 1, 1,
				0, 0, 0, 0,
				0, 1, 0, 0,
				0, 1, 1, 0};
		Tensor pred = Tensor({4,4}, p);
		CHECK(pred.isSame(y));
	}

	SECTION("Reverse Burst") {
		uint8_t in[] = {1, 1, 1, 1,
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 1, 1, 0,
				0, 0, 0, 0};
		Tensor x = Tensor({4,4}, in);
		Tensor y = defaultBackend()->reverseBurst(x);

		auto vec = y.toHost<uint8_t>();
		std::vector<uint8_t> pred_sum = {1,1,1,2,0};

		for(size_t i=0;i<4;i++) {
			size_t sum = 0;
			for(size_t j=0;j<4;j++)
				sum += vec[i*4+j];
			CHECK(sum == pred_sum[i]);
		}
	}

	SECTION("Grow Synapses") {
		int32_t synapses[4] = {0, 1, 1, -1};
		Tensor s = Tensor({2,2}, synapses);

		float perm[4] = {0.5, 0.4, 0.7, 0.0};
		Tensor p = Tensor({2,2}, perm);

		uint8_t in[2] = {1,1};
		Tensor x = Tensor({2}, in);
		Tensor y = Tensor({2}, in); //the same for test

		defaultBackend()->growSynapses(x, y, s, p, 0.21);
		defaultBackend()->sortSynapse(s, p);

		int32_t pred[] = {0,1 ,0,1};
		Tensor pred_conn = Tensor({2,2}, pred);

		float perms[] = {0.5, 0.4, 0.21, 0.7};
		Tensor pred_perm = Tensor({2,2}, perms);

		CHECK(s.isSame(pred_conn));
		CHECK(p.isSame(pred_perm));
	}
}

TEST_CASE("StateDict", "[StateDict]")
{
	using namespace et;
	StateDict state;

	Shape s = {4,5,6};
	state["key"] = s;
	CHECK(s.size() == 3);
	CHECK(state.size() == 1);
	CHECK_NOTHROW(state.at("key"));
	CHECK_THROWS(state.at("should_fail"));
	CHECK_NOTHROW(std::any_cast<Shape>(state["key"]));
	Shape s2 = std::any_cast<Shape>(state["key"]);
	CHECK(s == s2);
}

// TEST_CASE("Serealize")
// {
// 	using namespace et;

// }