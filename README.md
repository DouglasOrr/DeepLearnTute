# Deep Learning Tutorial

A simple setup for training MNIST in an interactive (Jupyter) notebook.

## Starting a session

First set up the workspace, adding some users:

    ./scripts/deploy prepare USER1 USER2 ...

Now start the server

    ./scripts/deploy start

You should now be able to go to `http://localhost` & log into your server
(for each account the password is just the username).

## Releasing

    # 1. Ensure code is committed & pushed (manual)

    # 2. Publish Docker image
    ./scripts/build
    docker tag deep-learn-tute douglasorr/deep-learn-tute:0.3
    docker tag deep-learn-tute douglasorr/deep-learn-tute:latest
    docker push douglasorr/deep-learn-tute:0.3
    docker push douglasorr/deep-learn-tute:latest

    # 3. Tag the release
    git push origin HEAD:refs/tags/0.3

    # 4. Increment version number in setup.py (manual), commit & push
