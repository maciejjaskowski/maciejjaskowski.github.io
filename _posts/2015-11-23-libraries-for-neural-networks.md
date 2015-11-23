---
title: Libraries for training and exercising Neural&nbsp;Networks
---

<p class="lead">There is quite a bit of tools out there to boost creation and training of Neural Networks. In this post I shortly describe some of them and explain when one should use them.</p>

Neural Networks have become very hot topic recently and me too, I got intrigued by the advertised capabilities of them. I hope a beginner like me two weeks ago might find this TL;DR of libraries useful and enlightening.


## High level libraries

One of the way one can work with neural networks on high level is working with layers. You basically decide that your network consist of e.g. Convolutional Layer, Fully Connected Layer and Softmax Layer, 
prescribe how to connect them and voila! you can start training your neural network and tweaking parameters. 

### PyBrain
One of such libraries is <a href="http://pybrain.org/docs/">PyBrain</a>. PyBrain, however isn't CUDA aware, so it will be slow. One other drawback is, it's not obvious how to share your trained network with someone.

### Lasagne
A promising alternative is <a href="http://lasagne.readthedocs.org/en/latest/">Lasagne</a>. Lasagne is based on Theano, therefore you can easily boost the training performance by using CUDA. I haven't checked out Lasagne yet but at first sight it looks similar to PyBrain in terms of API.

### Caffe
In many research papers, you will find references to <a href="http://caffe.berkeleyvision.org/">Caffe</a> and you can easily download weights of published networks and use them right away. Here, you define your network using ProtoBuf format, so you don't code anything, whatsover. Caffe supports CUDA. 


## Low level libraries

If you want to do something more fancy with your network you might want to checkout one of the libraries that help you build a graph of computations which is a byproduct of very nice property: these libraries calculate the derivaties (gradients) for you! In other words: fear not brackpropagation, as long as you know how to express your forward propagation as a graph. 

### Theano
The best known tool is <a href="http://deeplearning.net/software/theano/">Theano</a>. Storing and restoring a trained network is rather easy. It might become a PITA if you modify the way your network works between save and load.

If you want to train your network on CUDA (you do right?), you'll need to read the documentation carefully to understand memory management and tricks alike. 


### TensorFlow
<a href="http://www.tensorflow.org/">TensorFlow</a> is the library recently open sourced by Google. One nice thing of Theano is that you can make use of multiple GPU if you have them connected to a single machine. Haven't played with it yet, though.

# Summary
Unless you want to do research on Neural Networks themselves, you are better off starting with a high-level library for creating and training Neural Networks. 

I'll keep this post up to date as I go forward with my own "discoveries" in the field.
