# Deep Learning Tutorial

Setup for training a deep learning system for handwritten character recognition, using Chainer, in an interactive (Jupyter) notebook.

## The practical

For help with the practical, see our [DIY guide](doc/diy.md).

## Start your own server

As this is built on Docker, setup should be quite simple (on Linux & Mac):

 - Install [Docker](https://www.docker.com/products/overview)
 - Install [Python 3](https://www.python.org/downloads/)
 - Either clone this repo, or download the [deploy script](https://raw.githubusercontent.com/DouglasOrr/DeepLearnTute/master/scripts/deploy)

First set up the workspace, adding some users:

    ./scripts/deploy prepare USER1 USER2 ...

Now start the server:

    ./scripts/deploy start

You should now be able to go to http://localhost & log into your server (for each account you created the password is the same as the username).
Any files you save on the server should be accessible in `work/${USER}`.

### Windows

Most of this should work reasonably on Windows, however the "deploy" script makes use of Docker filesystem forwarding using `-v` (so that notebooks saved inside are visible outside the container). Your mileage may vary with this.
