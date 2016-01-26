---
title: Installing caffe MacOSX 10.11.2
---

<p class="lead"> <a href="http://jekyllrb.com">Jekyll</a> is a static site generator, an open-source tool for creating simple yet powerful websites of all shapes and sizes.</p>


> I upgraded my MacOSX to El Capitaine a couple of weeks ago but it's only now that problems started to pop out.



Upgrade XCode to 7 (7.2 in my case)

brew install openblas
brew install glog
brew install gflags
brew install boost
brew reinstall gcc
brew install leveldb
brew install lmdb


Wow!... http://caffe.berkeleyvision.org/install_osx.html

Reinstall Docker Toolbox -> 1.9.1i (because of Network timeoud during docker pull)
https://hub.docker.com/r/tleyden5iwx/caffe-cpu-master/
