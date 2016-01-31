---
layout: page
title: First Docker Container
---

The story begins with [Caffe](caffe.berkeleyvision.org), an excellent library for creating and training neural networks both on CPU and on GPU. The property particularly appealing to me is, that I can seimply download a pretrained Neural Network from Caffe [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) and adjust it to my needs(a.k.a. do transfer learning).

At first my plan was to install Caffe on my MacOSX. My heart didn't faint when I saw thin the installation guide on Caffe homepage 

> This route is not for the faint of heart. (...) If that is not an option, take a deep breath and carry on.

Actually, I was pretty advanced with the walkthrough but then I stumbled upon that quote:

> (...) Then, whenever you want to update homebrew, switch back to the master branches, do the update, rebase the caffe branches onto master and fix any conflicts.

No, thank you!

# Alternative way - Docker

I found this Docker Container out there and used it - this time, flawlessly. However, I desperately wanted to have Jupyter installed inside Docker, so that I can easily visualize images like this.

I will not show the whole walkthrough as you can simply download my Docker Container but I wanted to highlight how much fun it is to create your own container.

# First Dockerfile

Let's create a Dockerfile in some directory.

{% highlight python %}
FROM ubuntu:14.04

RUN apt-get install -y wget
RUN apt-get install -y bzip2

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-3.19.0-Linux-x86_64.sh && \
    /bin/bash /Miniconda2-3.19.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda2-3.19.0-Linux-x86_64.sh && \
    /opt/conda/bin/conda install --yes conda==3.19.0

{% endhighlight %}

Good! Let's try it out!



{% highlight bash %}
>  docker build --tag my/conda .
{% endhighlight %}

This should take a while as you need to download Ubuntu and then you should see sth like that:

{% highlight bash %}
Sending build context to Docker daemon 2.048 kB
Step 1 : FROM ubuntu:15.04
15.04: Pulling from library/ubuntu
fe9e3f6af4e1: Pull complete 
4516dd2e7bc0: Pull complete 
c1d1833b8e73: Pull complete 
99639e3e70c8: Pull complete 
Digest: sha256:2fb27e433b3ecccea2a14e794875b086711f5d49953ef173d8a03e8707f1510f
Status: Downloaded newer image for ubuntu:15.04
---> 99639e3e70c8o
Step 2 : RUN apt-get install -y wget
---> Running in 1edaf8b7a4a5
(...)
---> 8868ca69776f
Removing intermediate container 1edaf8b7a4a5
Step 3 : RUN apt-get install -y bzip2
 ---> Running in f660bb77a26e
(...)
---> 6b5d1d268411
Removing intermediate container f660bb77a26e
Step 4 : RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh &&     wget --quiet https://repo.continuum.io/miniconda/Miniconda2-3.19.0-Linux-x86_64.sh &&     /bin/bash /Miniconda2-3.19.0-Linux-x86_64.sh -b -p /opt/conda &&     rm Miniconda2-3.19.0-Linux-x86_64.sh &&     /opt/conda/bin/conda install --yes conda==3.19.0'
 ---> Running in 9800f5f5910e
{% endhighlight %}

Let's see what we've got installed

{% highlight bash %}
  docker run -it my/conda /opt/conda/bin/conda list
{% endhighlight %}

{% highlight bash %}
# packages in environment at /opt/conda:
#
conda                     3.19.0                   py27_0  
conda-env                 2.4.5                    py27_0  
openssl                   1.0.2f                        0  
pip                       8.0.1                    py27_0  
pycosat                   0.6.1                    py27_0  
pycrypto                  2.6.1                    py27_0  
python                    2.7.11                        0  
pyyaml                    3.11                     py27_1  
readline                  6.2                           2  
requests                  2.9.1                    py27_0  
setuptools                19.4                     py27_0  
sqlite                    3.9.2                         0  
tk                        8.5.18                        0  
wheel                     0.26.0                   py27_1  
yaml                      0.1.6                         0  
zlib                      1.2.8                         0  
{% endhighlight %}

It would be nice to add `/opt/conda/bin` to `$PATH`. Does it mean that I'll need to provision the whole container from scratch? Let's see!

# Extending Dockerfile
Let's add this line to the Dockerfile:

{% highlight python %}
  ENV PATH /opt/conda/bin:$PATH
{% endhighlight %}

{% highlight bash %}

docker build -t my/conda .

{% endhighlight %}

{% highlight bash %}
Sending build context to Docker daemon 2.048 kB
Step 1 : FROM ubuntu:15.04
 ---> 99639e3e70c8
Step 2 : RUN apt-get install -y wget
 ---> Using cache
 ---> 8868ca69776f
Step 3 : RUN apt-get install -y bzip2
 ---> Using cache
 ---> 6b5d1d268411
Step 4 : RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh &&     wget --quiet https://repo.continuum.io/miniconda/Miniconda2-3.19.0-Linux-x86_64.sh &&     /bin/bash /Miniconda2-3.19.0-Linux-x86_64.sh -b -p /opt/conda &&     rm Miniconda2-3.19.0-Linux-x86_64.sh &&     /opt/conda/bin/conda install --yes conda==3.19.0
 ---> Using cache
 ---> 02e0588fc948
Step 5 : ENV PATH /opt/conda/bin:$PATH
 ---> Running in 670b17a31f46
 ---> b7882879cb69
Removing intermediate container 670b17a31f46
Successfully built b7882879cb69
{% endhighlight %}

That was quick! If you track carefully the sha numbers you will see that Docker used intermediate containers (like 8868ca69776f and 6b5d1d268411)  that were build during the previous build of Dockerfile. Note that this works as long as you append to Dockerfile.

Let's take a look at these images:

{% highlight bash %}
docker images -a
{% endhighlight %}

{% highlight bash %}
my/conda               latest              b7882879cb69        3 minutes ago       287.3 MB
<none>                 <none>              02e0588fc948        14 minutes ago      287.3 MB
<none>                 <none>              6b5d1d268411        15 minutes ago      138.5 MB
<none>                 <none>              8868ca69776f        28 minutes ago      138.1 MB
{% endhighlight %}


# Summary
Docker is truely an amazing technology I will definitely exploit more often. 

There are a couple of caveats, though, that were driging me crazy: [Docker Machine issue 1](http://stackoverflow.com/questions/33442351/how-to-connect-to-a-service-running-on-docker-container-from-withing-host-macos/33467140?noredirect=1#comment57763764_33467140) and [internet connectivity issues](https://github.com/docker/docker/issues/13381) that one solves by restarting the docker daemon.
