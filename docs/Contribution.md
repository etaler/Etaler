# Contribution

There are may different ways to help!

## Using Etaler

The easiest and the most direct way to help is by using the library. If you find a bug or a missing feature, please open an issue and let us know how we can improve. If you have built a cool project usign Etaler. Share it!


## Developing Etaler

If you want to develop the library it self, you are very welcomed! We are excited to see a new PR pop up.

### Deisgn guidlines

* Don't care about x86-32 (Any 32bit architecture in fact). But no intentional breaks
* Use DOD. OOP is slow and evil
* Backend calls should be async-able, allow parallelization as much as possible
* Seprate the compute backend from the API frontend
* Make the design scalable
* See no reason to run a layer across multiple GPUs. Just keep layers running on a single device
* Keep the API high level
* Braking the archicture is OK
* Serealizable objects should reture a `StateDict` (`std::map<std::string, std::any>`) object for serialization
  * Non intrusive serialization
  * Reseves a StateDict object to deserealize
* Language binding should be in another repo
* Don't care about swarmming
* follow the KISS (Keep it Simple Stupid) principle
* Configuration files are evil (Looking at you NuPIC)

