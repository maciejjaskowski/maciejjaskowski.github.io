---
title: Choose your Neural Network Library
---

<p class="lead"></p>


## Tl;dr

| Library    	| Language   	| Documentation 	| Abstraction level             	| Extensibility 	| Open sourced by 	|
| [Caffe](caffe.berkeleyvision.org)     	| Python/C++ 	| poor          	| Layers (defined in txt files) 	| CuDNN         	| BerkeleyVision  	|
| [Lasagne](http://lasagne.readthedocs.org/)    	| Python     	| good          	| Layers (combined in Python)   	| Python and Theano 	| Community       	|
| [Keras](https://github.com/fchollet/keras/)         | Python        | good                  | Layers (combined in Python)           | Python and (Theano or TensorFlow) | Community |
| [Theano](https://github.com/Theano/Theano)     	| Python     	| good          	| Automatic differentiation     	| Python        	| Community       	|
| [TensorFlow](https://www.tensorflow.org/) 	| Python     	| (seems) good         	| Automatic differentiation     	| Python        	| Google          	|
| [Torch](https://github.com/torch/nn/)      	| Lua        	| (seems) good        	| Layers (defined in Lua) and Automatic differentiation        	| Lua           	| Facebook        	|

### Theano and TensorFlow
These are the "low level" general purpose libraries. Performance of Theano is for now superior over TensorFlow, however, TensorFlow has one fantastic feature not matched in Theano: [subgraph evaluation](https://www.quora.com/What-is-unique-about-Tensorflow-from-the-other-existing-Deep-Learning-Libraries).

Moreover, TensorFlow will eventually be adjusted so that you can train your neural network not only multiple GPU but also on mutiple machines with multiple GPUs.

### Caffe
Poor documentation is a fact. Learning curve is pretty steep and one gets lost in all the files you need to define to train your neural network. 

However, there are some [wonderful Python Notebooks with examples](https://github.com/BVLC/caffe/tree/master/examples) and is worth considering if you are to perform classic Neural Network tasks and want to quickly perform transfer learning based on some available networks from the [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo).

Among all the libraries working on with layers, this one feels the most closed one. Creating a new Layer, or even cost function is a pain that you can trace on their [Github issues](https://github.com/BVLC/caffe/issues) and it boils down to both C++ and CuDNN. 

Installing Caffe is also a big pain and I recommend either using available AMI with Caffe preinstalled or Docker ([anaconda based](https://hub.docker.com/r/mjaskowski/caffe-gpu/), [pip based](https://hub.docker.com/r/tleyden5iwx/caffe-gpu-master/))

### Keras
This one is tricky. At first sight it is tempting to use it because you can choose and change the underlying backend of Theano or TensorFlow. This feature, however, make it a thick framework that will be hard to extend because of the additional indirection level. Imagine you want to create a custom layer: that means that you not only need to understand how Theano (or TensorFlow) works but also how Keras works and uses Theano.

That all resembles too much JPA2 (or any other thick ORM), AngularJS (or any other thick UI framework) and ultimately it turns out to be no fun.

### Lasagne
This is the library I choose. It's build on top of Theano and extending is not harder then writing your own layers in Theano. In other words, once Lasagne will turn out to be not enough for me, I can easily fork Lasagne and fully adjust it to my needs. I had only one caveat with installation on MacOSX: I have to `export DYLD_FALLBACK_LIBRARY_PATH="$HOME/anaconda/envs/python27/lib"` as otherwise problems with loading `libmkl_intel_lp64.dylib` were reported.

### Torch
To use Torch, you must learn basics of Lua. However, it seems that Lua is like Theano and Lasagne combined. I will definitely try it out soon.

