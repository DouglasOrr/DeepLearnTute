# Deep Learning Tutorial

Setup for training a deep learning system for handwritten character recognition, using Chainer, in an interactive (Jupyter) notebook.

## The practical

For help with the practical, see our [DIY guide](doc/diy.md).

## Start your own server (selfhost)

This is built on top of Docker, which attempts to isolate the environment that your code is run in, so you shouldn't need to install anything except Docker on your computer:

 - Install [Docker](https://www.docker.com/community-edition)
 - Download the [selfhost script](https://raw.githubusercontent.com/DouglasOrr/DeepLearnTute/master/scripts/selfhost)
 - Run the selfhost script `chmod +x selfhost && ./selfhost`
   - Remember to download your files before stopping the server, as the server does not keep them
