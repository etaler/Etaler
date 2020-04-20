#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
//#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
#include <Etaler/Algorithms/SDRClassifer.hpp>
#include <Etaler/Utils/ProgressDisplay.hpp>
using namespace et;

#include <iostream>
#include <random>

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

/**
 * RESULTS:
 *
 * Order   : accuray    : boost         : SP size       : #epochs : comment
 * ------------------------------------------------------------------------------------------------------------------------------------------
 * 1       : 80.62%     : 9             : 16384         : 1       : more cells
 * 2       : 80.52%     : 9             : 8192          : 1       : initial setup
 * Baseline: 
 * Accuracy: 73.09%                     : SP disabled   : 1       : baseline with only classifier on raw images, no SP
 *
 */

static void printConfusionMatrix(const Tensor mat);
static void usage(const char *argv0);

int main(int argc, char** argv)
{
	// setDefaultBackend(std::make_shared<OpenCLBackend>()); // Run on the GPU!

	std::string data_path = ".";
	//Parse command line arguments
	if (argc == 2) {
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h") {
			usage(argv[0]);
			return 0;
		}
	}
	for (int count = 1; count < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--data_path"){
			if(count+1 <= argc)
				data_path = std::string(argv[count + 1]);
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\"" << std::endl;
			usage(argv[0]);
			return -1;
		}
	}

	//Read the MNIST dataset
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(data_path);
	if(dataset.training_labels.size() == 0) {
		std::cerr << "Cannot load MNIST dataset. from path: " << data_path << std::endl;
		return -1;
	}
	mnist::binarize_dataset(dataset);

	// HTM hyper parameters
	const size_t epochs = 1;
	const intmax_t sp_cells = 8192; // More cells, better accuracy
	const float global_density = 0.06; // slightly lesser than 0.1
	const float permanence_inc = 0.14; // HTM.core parameters
	const float permanence_dec = 0.006; // a lot lower than perm inc
	const float bootsting_factor = 9; // Some high enough value to promote expression

	// Other parameters
	const size_t display_steps = 100;

	// Create a Spatial Pooler with the paremeters we desire
	// Topology (aka SpatialPoolerND does not seem to do better job)
	SpatialPooler sp(Shape{28,28}, Shape{sp_cells});
	sp.setPermanenceDec(permanence_dec);
	sp.setPermanenceInc(permanence_inc);
	sp.setGlobalDensity(global_density);
	sp.setBoostingFactor(bootsting_factor);

	// HTM by itself cannot perform classifcation. Thus we need a classifer.
	// For NuPIC users, SDRClassifer in Eraler is CLAClassifer in NuPIC.
	SDRClassifer classifer(Shape{sp_cells}, 10);

	ProgressDisplay disp(dataset.training_images.size());

	// The traning happens in 2 stages. 1. We train the SP alone. Then 2. Train the SDRClassifer
	// Doing so ensures a stable SDR representation when traning the SDRClassifer. Otherwhise, a
	// learning SP will intoduce noice into the SDR classifer and lower the accuray

	// This is phase 1, we train the SP. In detail, we shuffle the dataset, train the SP and repeat
	// the process for a few times. Shuffling increases the randomness in the dataset and repeating
	// lets the SP to learn more.
	std::mt19937 rng;
	for(size_t i=0;i<epochs;i++) {
		std::cout << "Epoch " << i << "\n";
		// Shuffle the dataset
		std::vector<size_t> indices(dataset.training_images.size());
		for(size_t j=0;j<indices.size();j++)
			indices[j] = j;
		std::shuffle(indices.begin(), indices.end(), rng);
		
		// Train the Spatial Pooler
		for(size_t j=0;j<dataset.training_images.size();j++) {
			auto idx = indices[j];
			Tensor x = Tensor({28, 28}, dataset.training_images[idx].data());
			Tensor y = sp.compute(x);
			sp.learn(x, y);

			if(j%display_steps == 0)
				disp.update(j);
		}
		std::cout << "\n";
	}

	// Phase 2, we train the SDRClassifer. In this phase, we send whatever the SP generates into the 
	// classifer and tell the classifer what class the SDR belongs to. Since SDRClassifer is a lazy
	// learning algorithm, one iteration of traning is enough (it makes no difference doing more).
	// We also set boosting factor to 0 for a more stable SDR
	sp.setBoostingFactor(0);
	std::cout << "Traning SDRClassifer" << std::endl;
	disp.reset();
	for(size_t j=0;j<dataset.training_images.size();j++) {
		Tensor x = Tensor({28, 28}, dataset.training_images[j].data());
		Tensor y = sp.compute(x);

		classifer.addPattern(y, dataset.training_labels[j]);

		if(j%display_steps == 0)
			disp.update(j);
	}
	std::cout << "\n";

	// Test the model. We send testing images into the SP, then to the SDRClassifer. If the classifer
	// can classify the input correct, we're good!
	std::cout << "Testing model" << std::endl;
	disp = ProgressDisplay(dataset.test_images.size());
	size_t correct = 0;
	Tensor confusion_matrix = zeros({10, 10});
	for(size_t i=0;i<dataset.test_images.size();i++) {
		Tensor x = Tensor({28, 28}, dataset.test_images[i].data());
		Tensor y = sp.compute(x);

		intmax_t label = dataset.test_labels[i];
		intmax_t pred = classifer.compute(y);

		if(label == pred)
			correct += 1;

		if(i%display_steps == 0)
				disp.update(i);

		confusion_matrix[{label, pred}] = confusion_matrix[{label, pred}] + 1;
	}

	std::cout << std::endl;
	// std::cout << confusion_matrix.sum() << std::endl;

	printConfusionMatrix(confusion_matrix);

	std::cout << "Final accuracy: " << (float)correct/dataset.test_images.size()*100 << "%" << std::endl;

}

static void usage(const char *argv0)
{
	std::cout << "Usage: " << argv0 << " [--data_path path_to_dataset_folder]"
		<< std::endl;
}

// Some magic code that prints the confusion matrix
void printConfusionMatrix(const Tensor mat)
{
	const size_t spaces = 6;
	auto leftpad = [](std::string s, size_t n, char ch=' ') -> std::string {
		if(s.size() >= n)
			return s;
		std::string res = std::string(n-s.size(), ' ');
		return res + s;
	};
	Shape expected_shape = {10, 10};
	et_assert(mat.shape() == expected_shape);

	// Print the header
	std::cout << leftpad("", spaces+1);
	for(int j=0;j<10;j++)
		std::cout << leftpad(std::to_string(j), spaces) << " ";
	std::cout << std::endl;
	
	// Print each rows
	for(int i=0;i<10;i++) {
		std::cout << leftpad(std::to_string(i), spaces) << " ";
		for(int j=0;j<10;j++) {
			int v = mat.view({i, j}).item<int>();
			std::cout << leftpad(std::to_string(v), spaces) << " ";
		}
		std::cout << std::endl;
	}
}
