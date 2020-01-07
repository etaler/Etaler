#include <catch2/catch.hpp>

#include <Etaler/Etaler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
#include <Etaler/Encoders/Category.hpp>
#include <Etaler/Encoders/GridCell1d.hpp>
#include <Etaler/Encoders/GridCell2d.hpp>
#include <Etaler/Core/Serialize.hpp>
#include <Etaler/Algorithms/SDRClassifer.hpp>

#include <numeric>

using namespace et;

TEST_CASE("default backend sanity")
{
	REQUIRE(defaultBackend() != nullptr);
}

TEST_CASE("Testing Shape", "[Shape]")
{
	Shape s = {3,5};

	SECTION("Shape") {
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

	SECTION("Shape computation") {
		Shape stride = shapeToStride(s);
		CHECK(stride.size() == s.size());
		CHECK(stride == Shape({5,1}));

		Shape loc = {1,1};
		size_t idx = unfoldIndex(loc, s);
		CHECK(idx == 6);
		size_t idx2 = unfold(loc, stride);
		CHECK(idx2 == 6);

		CHECK(foldIndex(7, s) == Shape({1,2}));
	}

	SECTION("leftpad") {
		Shape s = {1,2,3};
		Shape t = leftpad(s, 4, 1);
		CHECK(t == Shape({1,1,2,3}));

		Shape q = leftpad(s, 1, 10);
		CHECK(q == Shape({1,2,3}));
	}
}

TEST_CASE("Testing Tensor", "[Tensor]")
{
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

	SECTION("Empty Tensor") {
		CHECK(Tensor().has_value() == false);
		int n = 1;
		CHECK(Tensor({1}, &n).has_value() == true);
	}

	SECTION("Create Tensor from scalar") {
		Tensor t = 7;
		CHECK(t.dtype() == DType::Int32);
		CHECK(t.shape() == Shape({1}));
		CHECK(t.item<int>() == 7);
		CHECK_THROWS(t.item<float>());

		Tensor q = 1.2f;
		CHECK(q.dtype() == DType::Float);
		CHECK(q.shape() == Shape({1}));
		CHECK((double)q.item<float>() == Approx(1.2));

		Tensor r = true;
		CHECK(r.dtype() == DType::Bool);
		CHECK(r.shape() == Shape({1}));
		CHECK(r.item<uint8_t>() == true);
	}

	SECTION("Create Tensor from vector") {
		std::vector<int> v = {1, 2, 3, 4};
		Tensor t = Tensor(v);
		CHECK(t.size() == intmax_t(v.size()));
		CHECK(t.dtype() == DType::Int);
	}

	SECTION("tensor like") {
		Tensor t = ones({4,4});
		Tensor q = ones_like(t);
		CHECK(q.dtype() == t.dtype());
		CHECK(q.shape() == t.shape());
		CHECK(q.backend() == t.backend());
		CHECK(q.isSame(t) == true);
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

		CHECK_THROWS(t.toHost<float>());

		auto data2 = t.toHost<int32_t>();
		Tensor q = Tensor({5}, data2.data());
		CHECK(t.isSame(q));

		Tensor r = t.copy();
		CHECK(t.isSame(r));
		CHECK(q.isSame(r));
	}

	SECTION("Property Check") {
		CHECK_NOTHROW(requireProperties(ones(Shape{1}, DType::Int32).pimpl(), DType::Int32));
		CHECK_THROWS(requireProperties(ones(Shape{1}, DType::Float).pimpl(), DType::Int32));
		CHECK_THROWS(requireProperties(ones(Shape{1}, DType::Float).pimpl(), IsDType{DType::Int32, DType::Bool}));

		CHECK_NOTHROW(requireProperties(ones(Shape{1}, DType::Int32).pimpl(), defaultBackend()));

		CHECK_NOTHROW(requireProperties(ones(Shape{1}, DType::Int32).pimpl(), IsContingous()));
		CHECK_THROWS(requireProperties(ones(Shape{4,4}, DType::Int32).view({range(2), range(2)}).pimpl(), IsContingous()));

		CHECK_NOTHROW(requireProperties(ones(Shape{1}, DType::Int32).pimpl(), IsPlain()));
		CHECK_NOTHROW(requireProperties(ones(Shape{1}, DType::Int32).view({0}).pimpl(), IsPlain()));
		CHECK_THROWS(requireProperties(ones(Shape{4,4}, DType::Int32).view({range(2), range(2)}).pimpl(), IsPlain()));
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
			CHECK(r.shape() == Shape({2,2}));
			int a[] = {0,1,4,5};
			Tensor pred = Tensor({2,2}, a);
			CHECK(realize(r).isSame(pred));
		}

		SECTION("Indexing a column") {
			Tensor q = t.view({all(), 0});
			int arr[] = {0, 4, 8, 12};
			Tensor r = Tensor({4}, arr);
			CHECK(q.shape() == Shape({4}));
			CHECK(r.isSame(q));
		}

		SECTION("Indexing with negative values") {
			Tensor q = t.view({3});
			Tensor r;
			CHECK_NOTHROW(r = t.view({-1}));
			CHECK(r.has_value() == true);
			CHECK(q.isSame(r));
		}

		SECTION("View of views") {
			Tensor t = ones({4, 4});
			Tensor v1 = t[{3}];
			Tensor v2 = v1[{all()}];
			CHECK(v2.size() == 4);
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

			t.view({range(2),range(2)}) = constant({2,2}, 2);
			int b[] = {2,2,2,3
				,2,2,6,7
				,8,9,10,11
				,12,13,14,15};
			Tensor pred2 = Tensor({4,4}, b);
			CHECK(t.isSame(pred2));

			Tensor s = t.view({range(2),range(2)});
			s = constant({2,2}, 3); //Should change nothing
			CHECK(t.isSame(pred2));

			//Assign from view
			Tensor u = ones({4,4});
			u.view({all(), all()}) = zeros({1});
			CHECK(u.isSame(zeros_like(u)));
		}

		SECTION("Strided views") {
			SECTION("read") {
				int a[] = {0, 2};
				Tensor q = Tensor({2}, a);
				Tensor res = t.view({0, range(0, 3, 2)});
				CHECK(res.isSame(q));
			}

			SECTION("write") {
				int a[] = {-1, -1};
				Tensor q = Tensor({2}, a);
				t[{0, range(0, 3, 2)}] = q;

				int b[] = {-1, 1, -1, 3};
				Tensor r = Tensor({4}, b);
				CHECK(t[{0}].isSame(r));
			}
		}

		SECTION("subscription operator") {
			IndexList r = {range(2)};
			//The [] operator should work exactly like the view() function
			CHECK(t[r].isSame(t.view(r)));
		}

		SECTION("assign to subscription (scalar)") {
			t[{2, 2}] = t[{2, 2}] + 1;
			CHECK(t[{2, 2}].item<int>() == 11);
		}

		SECTION("assign to subscription (vector)") {
			t[{2}] = t[{2}] + 1;
			//Check a subset of weather the result is correct
			CHECK(t[{2, 2}].item<int>() == 11);
		}

		SECTION("all/any test") {
			CHECK(ones({3}).any() == true);
			CHECK(zeros({3}).any() == false);
			CHECK(ones({3}).all() == true);
			CHECK(zeros({3}).all() == false);
			CHECK((ones({7}) == zeros({7})).all() == false);
			CHECK((ones({7}) == zeros({7})).any() == false);
			CHECK((ones({4,4}) == t).any() == true);
			CHECK((ones({4,4}) == t).all() == false);
		}
	}

	SECTION("item") {
		Tensor t = ones({1});
		CHECK(t.item<int>() == 1);
		// item() should fail because asking for the wrong type
		CHECK_THROWS(t.item<float>());
		
		Tensor q = ones({2});
		// item() should fail because q is not a scalar
		CHECK_THROWS(q.item<int>());
	}

	SECTION("iterator") {
		Tensor t = ones({3, 4});
		Tensor q = zeros({3, 4});
		STATIC_REQUIRE(std::is_same_v<Tensor::iterator::value_type, Tensor>);

		// Tensor::iterator should be ramdp,
		// Reference: http://www.cplusplus.com/reference/iterator/RandomAccessIterator/
		STATIC_REQUIRE(std::is_default_constructible_v<Tensor::iterator>);
		STATIC_REQUIRE(std::is_copy_constructible_v<Tensor::iterator>);
		STATIC_REQUIRE(std::is_copy_assignable_v<Tensor::iterator>);
		STATIC_REQUIRE(std::is_destructible_v<Tensor::iterator>);
		CHECK(t.begin() != t.end());
		CHECK(t.begin() == t.begin());
		CHECK((*t.begin()).shape() == Shape{4});
		CHECK(t.begin()->shape() == Shape{4});
		CHECK(t.end() - t.begin() == t.shape()[0]);
		CHECK(t.begin()[2].isSame(*t.back()) == true);
		auto it1 = t.begin(), it2 = t.begin();
		it1++;
		++it2;
		CHECK(it1 == it2);
		--it1;
		it2--;
		CHECK(it1 == it2);

		swap(*t.begin(), *q.begin());
		CHECK(t[{0}].isSame(zeros({4})));

		int num_iteration = 0;
		for(auto s : t) {
			CHECK(s.shape() == Shape({4}));
			s.assign(constant({4}, 42));
			num_iteration += 1;
		}
		CHECK(num_iteration == t.shape()[0]);
		CHECK(t.sum().item<int>() == 42*t.size());
	}

	SECTION("swapping Tensor") {
		std::vector<int> v1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12};
		std::vector<int> v2 = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20 , 21, 22};
		Tensor t = Tensor(v1).reshape({3, 4});
		Tensor q = Tensor(v2).reshape({3, 4});
		Tensor t_old = t.copy();
		Tensor q_old = q.copy();

		SECTION("swap") {
			swap(t, q);
			CHECK(t.isSame(q_old));
			CHECK(q.isSame(t_old));
		}

		SECTION("swapping views") {
			swap(q[{1}], t[{2}]);
			CHECK(t[{2}].isSame(q_old[{1}]));
			CHECK(q[{1}].isSame(t_old[{2}]));
		}

		SECTION("swaping itself") {
			swap(t[{0}], t[{0}]);
			REQUIRE(t[{0}].isSame(t_old[{0}]));

			swap(t[{0}], t[{1}]);
			REQUIRE(t[{0}].isSame(t_old[{1}]));
			REQUIRE(t[{1}].isSame(t_old[{0}]));
		}
	}
}

TEST_CASE("Testing Encoders", "[Encoder]")
{
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

		auto categories = decoder::category(t, num_categories);
		CHECK(categories.size() == 1);
		CHECK(categories[0] == 0);
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

		//GridCell encoders should have a very small amount of bits overlaping
		t = encoder::gridCell1d(0.1, 16, 1, 16);
		q = encoder::gridCell1d(0.5, 16, 1, 16);
		CHECK((t&&q).sum().toHost<int>()[0] < 16*0.4);
	}

	SECTION("GridCell2d Encoder") {
		Tensor t = encoder::gridCell2d({0.1, 0.1}, 16, 1, {4,4});
		CHECK(t.size() == 16*4*4);
		auto v = t.toHost<uint8_t>();
		CHECK(std::accumulate(v.begin(), v.end(), 0) == 16);

		Tensor q = encoder::gridCell2d({0.1, 0.1}, 16, 2, {4,4});
		CHECK(q.size() == 16*16);
		auto u = q.toHost<uint8_t>();
		CHECK(std::accumulate(u.begin(), u.end(), 0) == 32);

		//GridCell encoders should have a very small amount of bits overlaping
		t = encoder::gridCell2d({0.1, 0.3}, 16, 1);
		q = encoder::gridCell2d({10, 30}, 16, 1);
		CHECK((t&&q).sum().toHost<int>()[0] < 16*0.4);
	}
}

TEST_CASE("Backend functions", "[Backend]")
{
	SECTION("Cell Activity") {
		int32_t synapses[4] = {0, 1, 1, -1};
		Tensor s = Tensor({2,2}, synapses);

		float perm[4] = {0.5, 0.4, 0.7, 0.0};
		Tensor p = Tensor({2,2}, perm);

		uint8_t in[2] = {1,1};
		Tensor x = Tensor({2}, in);

		Tensor y = cellActivity(x, s, p, 0.1, 1);
		CHECK(y.size() == 2);

		auto res = y.toHost<int32_t>();
		REQUIRE(res.size() == 2);
		CHECK(res[0] == 2);
		CHECK(res[1] == 1);
	}

	SECTION("Global Inhibition") {
		int32_t in[8] = {0,0,1,2,7,6,5,3};
		Tensor t = Tensor({8}, in);

		Tensor y = globalInhibition(t, 0.5);
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

		sortSynapse(t, q);

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
		Tensor y = burst(x, state);

		uint8_t p[] = {1, 1, 1, 1,
				0, 0, 0, 0,
				0, 1, 0, 0,
				0, 1, 1, 0};
		Tensor pred = Tensor({4,4}, p);
		CHECK(pred.isSame(y));
	}

	SECTION("Reverse Burst") {
		uint8_t in[] = {1, 1, 1, 1, // This column should be set to only 1 active cell
				1, 0, 0, 0, // Other shouldn't
				0, 1, 0, 0,
				0, 1, 1, 0,
				0, 0, 0, 0};
		Tensor x = Tensor({5,4}, in);
		Tensor y = reverseBurst(x);

		std::vector<int> pred_sum = {1, 1, 1, 2, 0};
		Tensor p = Tensor({5}, pred_sum.data());

		CHECK(y.sum(1).isSame(p));
	}

	SECTION("Grow Synapses") {
		int32_t synapses[4] = {0, 1, 1, -1};
		Tensor s = Tensor({2,2}, synapses);

		float perm[4] = {0.5, 0.4, 0.7, 0.0};
		Tensor p = Tensor({2,2}, perm);

		uint8_t in[2] = {1,1};
		Tensor x = Tensor({2}, in);
		Tensor y = Tensor({2}, in); //the same for test

		growSynapses(x, y, s, p, 0.21);
		sortSynapse(s, p);

		int32_t pred[] = {0,1 ,0,1};
		Tensor pred_conn = Tensor({2,2}, pred);

		float perms[] = {0.5, 0.4, 0.21, 0.7};
		Tensor pred_perm = Tensor({2,2}, perms);

		CHECK(s.isSame(pred_conn));
		CHECK(p.isSame(pred_perm));
	}

	SECTION("sum") {
		std::vector<int> v(16);
		std::iota(v.begin(), v.end(), 0);
		Tensor t = Tensor({4,4}, v.data());

		auto res = sum(t).toHost<int32_t>();
		CHECK(res.size() == 1);
		CHECK(res[0] == 120);

		Tensor s0 = sum(t, 0);
		CHECK(s0.size() == 4);
		CHECK(s0.dimentions() == 1);
		CHECK(s0.dtype() == DType::Int32);
		int32_t pred0[] = {24, 28, 32, 36};
		CHECK(s0.isSame(Tensor({4}, pred0)));

		Tensor s1 = sum(t, 1);
		CHECK(s1.size() == 4);
		CHECK(s1.dimentions() == 1);
		CHECK(s1.dtype() == DType::Int32);
		int32_t pred1[] = {6, 22, 38, 54};
		CHECK(s1.isSame(Tensor({4}, pred1)));
	}

	SECTION("sum with negative index") {
		auto a = ones({1, 2, 3});
		CHECK(sum(a, -1).shape() == Shape({1, 2}));
		CHECK(sum(a, -2).shape() == Shape({1, 3}));
		CHECK(sum(a, -3).shape() == Shape({2, 3}));

		CHECK_THROWS(sum(a, -4).shape());
	}

	SECTION("decay synapses") {
		int a[] = {0,1,0,1};
		Tensor c({2,2}, a);

		float b[] = {0.1, 0.7, 0.5, 0.01};
		Tensor p({2,2}, b);

		decaySynapses(c, p, 0.2);

		int pred[] = {1, -1, 0, -1};
		CHECK(Tensor({2,2}, pred).isSame(c));

		CHECK((double)p[{0, 0}].item<float>() == Approx(0.7).epsilon(1.e-5));
		CHECK((double)p[{1, 0}].item<float>() == Approx(0.5).epsilon(1.e-5));
		// Only values from good synapses are defined. The others can be erases, set
		// to 0, etc. Depending on implementation.
	}
}

TEST_CASE("StateDict", "[StateDict]")
{
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

TEST_CASE("Tensor operations")
{
	SECTION("cast") {
		int arr[] = {0, 768, 200, 40};
		Tensor t = Tensor({4}, arr);
		Tensor q = t.cast(DType::Bool);

		uint8_t pred[] = {0, 1, 1, 1};
		Tensor p = Tensor({4}, pred);
		CHECK(q.shape() == Shape({4}));
		CHECK(q.isSame(p));
	}

	SECTION("Unary operation") {
		int arr[] = {0,1,2,3};
		Tensor a = Tensor({4}, arr);

		SECTION("negate") {
			Tensor b = -a;
			CHECK(b.size() == a.size());
			CHECK(b.dtype() == a.dtype());
			int pred[] = {0, -1, -2, -3};
			Tensor p = Tensor({4}, pred);
			CHECK(b.isSame(p));
		}

		SECTION("logical_not") {
			Tensor b = !a;
			CHECK(b.size() == 4l);
			CHECK(b.dtype() == DType::Bool);
			bool pred[] = {1, 0, 0, 0};
			Tensor p = Tensor({4}, pred);
			CHECK(b.isSame(p));
		}
	}

	SECTION("Binary Operations") {
		int arr[] = {1,2,3};
		Tensor a = Tensor({3}, arr);

		SECTION("add") {
			Tensor b = a + a;
			CHECK(b.shape() == Shape({3}));
			CHECK(b.dtype() == DType::Int32);
			int pred[] = {2,4,6};
			Tensor p = Tensor({3}, pred);
			CHECK(b.isSame(p));
		}

		SECTION("subtract") {
			Tensor b = a - a;
			CHECK(b.shape() == Shape({3}));
			CHECK(b.dtype() == DType::Int32);
			int pred[] = {0,0,0};
			Tensor p = Tensor({3}, pred);
			CHECK(b.isSame(p));
		}

		SECTION("mul") {
			Tensor b = a * a;
			CHECK(b.shape() == Shape({3}));
			CHECK(b.dtype() == DType::Int32);
			int pred[] = {1,4,9};
			Tensor p = Tensor({3}, pred);
			CHECK(b.isSame(p));
		}

		SECTION("div") {
			Tensor b = a / a;
			CHECK(b.shape() == Shape({3}));
			CHECK(b.dtype() == DType::Int32);
			int pred[] = {1,1,1};
			Tensor p = Tensor({3}, pred);
			CHECK(b.isSame(p));
		}

		SECTION("equal") {
			Tensor b = a == a;
			CHECK(b.shape() == Shape({3}));
			CHECK(b.dtype() == DType::Bool);
			bool pred[] = {1,1,1};
			Tensor p = Tensor({3}, pred);
			CHECK(b.isSame(p));
		}

		SECTION("concat") {
			Tensor a = ones({2, 2});
			Tensor b = zeros({2, 2});
			Tensor c = zeros({2, 2}, DType::Float);
			Tensor d = zeros({2, 3});
			CHECK_THROWS(cat({a, c}));
			
			Tensor sol1 = ones({4, 2});
			sol1.view({range(2, 4), all()}) = 0;
			CHECK(cat({a, b}).isSame(sol1));

			Tensor sol2 = ones({2, 5});
			sol2.view({all(), range(2, 5)}) = 0;
			CHECK(cat({a, d}, /*dim=*/1).isSame(sol2));

			CHECK_THROWS(cat({a, d}));
		}
	}
}

TEST_CASE("brodcast")
{
	Tensor a, b;
	SECTION("no brodcast") {
		a = ones({4});
		b = ones({4});
		CHECK((a+b).shape() == Shape({4}));
	}

	SECTION("simple brodcast") {
		a = ones({2, 4});
		b = ones({4});
		CHECK((a+b).shape() == Shape({2, 4}));
	}

	SECTION("brodcast from {1}") {
		a = ones({2, 4});
		b = ones({1});
		CHECK((a+b).shape() == Shape({2, 4}));
	}

	SECTION("brodcast with a 1 axis") {
		a = ones({2, 1, 4});
		b = ones({2, 5, 4});
		CHECK((a+b).shape() == Shape({2, 5, 4}));
	}

	SECTION("bad brodcasting") {
		a = ones({2, 4});
		b = ones({7});
		CHECK_THROWS(a+b);
	}
}

TEST_CASE("SDRClassifer")
{
	size_t num_bits = 8;
	size_t num_category = 6;
	SDRClassifer classifer({(intmax_t)(num_bits*num_category)}, num_category);

	for(size_t i=0;i<num_category;i++)
		CHECK_NOTHROW(classifer.addPattern(encoder::category(i, num_category, num_bits), i));
	
	for(size_t i=0;i<num_category;i++)
		CHECK(classifer.compute(encoder::category(i, num_category, num_bits),0) == i);
}

TEST_CASE("Type system")
{
	SECTION("Type sizes") {
		STATIC_REQUIRE(dtypeToSize(DType::Bool) == 1);
		STATIC_REQUIRE(dtypeToSize(DType::Int32) == 4);
		STATIC_REQUIRE(dtypeToSize(DType::Float) == 4);
		STATIC_REQUIRE(dtypeToSize(DType::Half) == 2);
	}

	SECTION("type to dtype") {
		STATIC_REQUIRE(typeToDType<int32_t>() == DType::Int32);
		STATIC_REQUIRE(typeToDType<float>() == DType::Float);
		STATIC_REQUIRE(typeToDType<bool>() == DType::Bool);
		STATIC_REQUIRE(typeToDType<half>() == DType::Half);
	}

	SECTION("type of Tensor operatoins") {
		bool support_fp16 = [&](){
			try {ones({1}, DType::Half);}
			catch(const EtError&) {return false;}
			return true;
		}();

		SECTION("abs") {
			CHECK(abs(ones({1}, DType::Bool)).dtype() == DType::Int32);
			CHECK(abs(ones({1}, DType::Int32)).dtype() == DType::Int32);
			CHECK(abs(ones({1}, DType::Float)).dtype() == DType::Float);
			if(support_fp16)
				CHECK(abs(ones({1}, DType::Half)).dtype() == DType::Half);
		}

		SECTION("exp") {
			CHECK(exp(ones({1}, DType::Bool)).dtype() == DType::Float);
			CHECK(exp(ones({1}, DType::Int32)).dtype() == DType::Float);
			CHECK(exp(ones({1}, DType::Float)).dtype() == DType::Float);
			if(support_fp16)
				CHECK(exp(ones({1}, DType::Half)).dtype() == DType::Half);
		}

		SECTION("negation") {
			CHECK((-ones({1}, DType::Bool)).dtype() == DType::Int32);
			CHECK((-ones({1}, DType::Int32)).dtype() == DType::Int32);
			CHECK((-ones({1}, DType::Float)).dtype() == DType::Float);
			if(support_fp16)
				CHECK((-ones({1}, DType::Half)).dtype() == DType::Half);
		}

		SECTION("inverse") {
			CHECK(inverse(ones({1}, DType::Bool)).dtype() == DType::Float);
			CHECK(inverse(ones({1}, DType::Int32)).dtype() == DType::Float);
			CHECK(inverse(ones({1}, DType::Float)).dtype() == DType::Float);
			if(support_fp16)
				CHECK(inverse(ones({1}, DType::Half)).dtype() == DType::Half);
		}

		SECTION("log") {
			CHECK(log(ones({1}, DType::Bool)).dtype() == DType::Float);
			CHECK(log(ones({1}, DType::Int32)).dtype() == DType::Float);
			CHECK(log(ones({1}, DType::Float)).dtype() == DType::Float);
			if(support_fp16)
				CHECK(log(ones({1}, DType::Half)).dtype() == DType::Half);
		}

		SECTION("logical_not") {
			CHECK(logical_not(ones({1}, DType::Bool)).dtype() == DType::Bool);
			CHECK(logical_not(ones({1}, DType::Int32)).dtype() == DType::Bool);
			CHECK(logical_not(ones({1}, DType::Float)).dtype() == DType::Bool);
			if(support_fp16)
				CHECK(logical_not(ones({1}, DType::Half)).dtype() == DType::Bool);
		}

		SECTION("sum") {
			CHECK(ones({1}, DType::Bool).sum().dtype() == DType::Int32);
			CHECK(ones({1}, DType::Int32).sum().dtype() == DType::Int32);
			CHECK(ones({1}, DType::Float).sum().dtype() == DType::Float);
			if(support_fp16)
				CHECK(ones({1}, DType::Half).sum().dtype() == DType::Half);
		}

		auto solve_binary_op_type = [](DType self, DType other)->DType {
			// Implement a C++ like type promotion rule
			if(other == DType::Float || self == DType::Float)
					return DType::Float;
				else if(other == DType::Half || self == DType::Half)
					return DType::Half;
				return DType::Int32; //Even bool is promoted to int in operation
		};

		std::vector<DType> types = {DType::Int32, DType::Bool, DType::Float, DType::Half};
		SECTION("add") {
			for(auto t1 : types) {
				for(auto t2 : types) {
					if((t1 == DType::Half || t2 == DType::Half) && support_fp16 == false)
						continue;
					Tensor t = ones({1}, t1);
					Tensor q = ones({1}, t2);
					CHECK((t+q).dtype() == solve_binary_op_type(t1, t2));
				}
			}
		}

		SECTION("subtract") {
			for(auto t1 : types) {
				for(auto t2 : types) {
					if((t1 == DType::Half || t2 == DType::Half) && support_fp16 == false)
						continue;
					Tensor t = ones({1}, t1);
					Tensor q = ones({1}, t2);
					CHECK((t-q).dtype() == solve_binary_op_type(t1, t2));
				}
			}
		}

		SECTION("mul") {
			for(auto t1 : types) {
				for(auto t2 : types) {
					if((t1 == DType::Half || t2 == DType::Half) && support_fp16 == false)
						continue;
					Tensor t = ones({1}, t1);
					Tensor q = ones({1}, t2);
					CHECK((t*q).dtype() == solve_binary_op_type(t1, t2));
				}
			}
		}

		SECTION("div") {
			for(auto t1 : types) {
				for(auto t2 : types) {
					if((t1 == DType::Half || t2 == DType::Half) && support_fp16 == false)
						continue;
					Tensor t = ones({1}, t1);
					Tensor q = ones({1}, t2);
					CHECK((t/q).dtype() == solve_binary_op_type(t1, t2));
				}
			}
		}
	}
}

// TODO: Should I count this as an integration test?
// This test checks all components of Tensor works together properly
TEST_CASE("Complex Tensor operations")
{
	std::vector<int> v1 = {1, 8, 6, 7
			, 3, 2, 5, 6
			, 4, 3, 2, 7
			, 9, 0 ,1, 1};
	Tensor a = Tensor(v1).reshape({4,4});

	SECTION("Vector inner product") {
		std::vector<int> v1 = {1, 6, 7, 9, 15, 6};
		std::vector<int> v2 = {3, 7, 8, -1, 6, 15};
		REQUIRE(v1.size() == v2.size());
		Tensor a = Tensor(v1);
		Tensor b = Tensor(v2);

		CHECK((a*b).sum().item<int>() == std::inner_product(v1.begin(), v1.end(), v2.begin(), 0));
	}

	SECTION("assign column to row") {
		std::vector<int> v2 = {9, 0, 1, 1};
		Tensor b = Tensor(v2);

		Tensor t = a.copy();
		t[{all(), 1}] = a[{3}];
		CHECK(t[{all(), 1}].isSame(b));
	}

	SECTION("shuffle") {
		std::mt19937 rng;
		std::shuffle(a.begin(), a.end(), rng);
		CHECK(std::accumulate(v1.begin(), v1.end(), 0) == a.sum().item<int>());
	}

	SECTION("find_if") {
		Tensor b = a[{0}];
		CHECK(std::find_if(a.begin(), a.end(), [&b](auto t){ return t.isSame(b); }) != a.end());
	}

	SECTION("transform") {
		Tensor a = ones({12, 6});
		Tensor b = ones({12, 6});
		std::transform(a.begin(), a.end(), b.begin(), [](const auto& t){return zeros_like(t);});
		CHECK(b.isSame(zeros_like(a)));
	}

	SECTION("accumulate") {
		// Test summing along the first dimension. Making sure iterator and sum() works
		// Tho you should always use the sum() function instead of accumulate or reduce
		Tensor t = std::accumulate(a.begin(), a.end(), zeros({a.shape()[1]}));
		Tensor a_sum = a.sum(0);
		CHECK(t.isSame(a_sum));
	}

	SECTION("generate") {
		// Again, you should never do this. But you should be able to.
		Tensor t  = Tensor({4, 4}, DType::Int);
		std::generate(t.begin(), t.end(), [n = 0, &v1]() mutable {return Tensor({4}, v1.data()+(n++)*4);});
		CHECK(t.isSame(a));
	}
}

// TEST_CASE("Serealize")
// {
// 	using namespace et;

// }
