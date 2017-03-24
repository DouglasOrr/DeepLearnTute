# Do-It-Yourself Deep Learning Tutorial in Chainer

We want to encourage you to do-it-yourself in this practical. But to make it achievable in a couple of hours, we have done the following for you:

 - Chosen a problem to solve & found a dataset.
 - Provided code for loading and displaying the dataset.
 - Set up a machine for development.
 - Suggested a series of exercises to guide you through creating the model (below).

If you follow through the _**Exercise**_ lines as you go, you should end up with a simple, working, deep learning system.


## 0. Preparation

To become familiar with some of the libraries we'll be using, have a look at the `Intro.ipynb` notebook - this contains a few exercises to get you started.
Once you're happy with that, open `Tutorial.ipynb`, and start following through this worksheet.

_**Tip** - remember to keep saving your notebook, and consider downloading it at the end_


## 1. The Problem

Imagine we want to make a system that allows entering hand-drawn text on a tablet or smartphone. One key part of this will be recognizing individual characters (letters and numbers) - this will be our problem for today.

Formally, this problem is _multiclass classification_ (within _supervised learning_). The thing you would like to predict (often called _target_) is a one-of-N _label_ (often represented as an integer `[0 N)`). The 62 labels we would like to predict are:

    0 1 2 3 4 5 6 7 8 9
    A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
    a b c d e f g h i j k l m n o p q r s t u v w x y z

When we define a problem, it is also useful to think of an _objective_ - this is a single number which measures how good we are at solving the problem.
Also called the _loss_, this is something we should minimize in order to find good a good deep learning function.

We strongly recommend using the loss called `softmax_cross_entropy` for this exercise. This loss function accepts a score for each _target label_, and is smaller (better) when the actual target has a higher score than other targets.

_**Exercise** - use the `test_softmax_cross_entropy` function to answer the following questions:_
 - what scores `[a, b]` give the best (lowest) softmax_cross_entropy for balanced coin tosses (targets: `[0, 1]`)?
   - what scores give the worst (highest) softmax_cross_entropy?
 - what scores give the best softmax_cross_entropy for unbalanced coin tosses (targets: `[1, 1, 1, 1, 1]`)?
 - for a dice roll (6 scores), if you predict `[0, 0, 0, 0, 0, 0]`, what is the softmax_cross_entropy of the following rolls: `[0, 0, 1, 2, 3, 4, 5, 5, 5, 5]`?
   - what if you predict `[1, 1, 1, 1, 1, 1]`?
   - how can you get a better softmax_cross_entropy?

Hopefully now you have some intuition for how softmax_cross_entropy behaves - it prefers high scores for targets that are frequent (likely to occur), and low scores for targets that are infrequent (unlikely to occur). This makes it a good loss function for multiclass classification, so to solve our problem we must try to minimize softmax_cross_entropy.


## 2. Getting Data

Our data is drawn from the [UJI Pen Characters v2 Data Set](https://archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters+(Version+2)). Some key stats:

 - Number of instances: 11640
 - Number of labels: 97
 - Number of writers: 60

The original data is a sequence of strokes (xy coordinate pairs for a stylus on an electronic input device).

We have preprocessed the data, by:

 - Removing unwanted labels (accented characters & symbols)
 - Separating randomly into training, validation & test sets
    - N.B. this was done slightly sub-optimally, as each sample was considered independently for inclusion into train or test. The training data therefore contains letters that were entered by the same writer as letters in the validation and test sets. It would be better to separate out certain writers entirely for validation/test.
 - Augmenting training data - scale & rotate each training example to generate more training examples.
 - Normalizing & rendering to 16x16 greyscale patches - it is easier to use these as input.

After that, the data looks like:

    train = dlt.load_hdf5('/data/uji/train.hdf')
    print(train.x.shape)  # (93000, 256)
    print(train.y.shape)  # (93000,)

    print(train.x[0, :])           # [ 0.00 0.00 ... ] - a single example (256-vector of floats [0 1])
    print(train.y[0])              # 21 - label representation
    print(train.vocab[train.y[0]]) # L  - actual label

_**Exercise** - use slicing to select 20 consecutive images (xs), and 20 associated labels (ys), without writing a loop._

Now might be a good time to think - if you weren't using deep learning, how would you solve this problem - how would you figure out the ys from just the xs?


## 3. Defining a Model

A deep learning function is a parameterized function which maps input features to an output, which can be compared with an ideal output using a loss function.

In general, we define the deep learning function for _batches_ of data (since it is typically computationally inefficient to process just one example at a time). For this reason, the first dimension in the shape of the inputs & outputs is the "batch dimension" (call it `N`).

Our input is as described above - a greyscale image represented as a 256-element vector of floats in the range `[0 1]`. Due to batching, the input shape is therefore `(N, 256)`.

Our output will be a set of scores for each label (an array of `62` float scores), which can be fed into the `softmax_cross_entropy` loss. There will be a different set of scores for each example in the batch (as the scores must depend on the input image), so the output shape is `(N, 62)`.

So we need to design a function that maps `(N, 256) -> (N, 62)`.

### Chainer

To express this function, we'll use our framework - Chainer. Here is a review of the basics (see also the [tutorial](http://docs.chainer.org/en/stable/tutorial/basic.html#forward-backward-computation)):

    import chainer as C

    # Get data from numpy into a Variable
    # (N.B. you may often have to use np.float32/int32 explicitly)
    x = C.Variable(np.zeros((10, 3), dtype=np.float32))

    # Get data back out to numpy
    print(x.data)

    # Work on Variables using Functions (y & z are also Variables)
    y = x + 2                 # shape: (10, 3)
    z = C.functions.tanh(y)   # shape: (10, 3)

    # Create a Link, which holds parameters & acts like a function
    transform = C.links.Linear(3, 5)
    out = transform(x)                # shape: (10, 5)

_**Exercise** - use `C.links.Linear` to map a variable of shape `(20, 256) -> (20, 62)`, and run it on a batch of images from the input data `train.x`._

_**Note** - having done this, you have defined a simple network that could solve this task!_


## 4. Training the Model

We're almost ready to train our network. To train a network, it must be contained within a single `Link` - fortunately, we only have a single link in our first function, so we're already there.

> Note: This is a very condensed version of the [Chainer tutorial](http://docs.chainer.org/en/stable/tutorial/basic.html#optimizer) - see that if you get stuck.

Training a network consists of taking a sequence of steps to decrease a loss function, based on small batches of the training data. We'll show you how to take one such step - it'll be up to you to code up a training loop based on this.

    link = ...  # my model

    # First time setup (only once): choose an optimizer & set it up
    opt = C.optimizers.SGD()   # start with SGD, but I'd recommend trying Adam soon!
    opt.use_cleargrads()
    opt.setup(link)

    # For each batch: run the network, and take a single small gradient-based step
    x = C.Variable(...)   # shape (N, 256)
    y = C.Variable(...)   # shape (N,) integers
    link.cleargrads()     # clear gradients
    p = link(x)           # run the model to get predictions (shape (N, 62))
                          # (this should include any other functions you're using)
    loss = C.functions.softmax_cross_entropy(p, y)
    loss.backward()       # compute gradients
    opt.update()          # update parameters
    print(loss.data)      # how well are we doing, smaller is better

_**Exercise** - take a single optimization step for the network you defined earlier, using a batch of actual data._

Now this needs to become a training loop. A classic training loop will pass through the training data multiple times, splitting it up into batches for individual steps (each taken as above).

_**Exercise** - turn your code for a single step into a full training loop, for batch size 128, passing through **all** of the training data 4 times._
  - _**Hint** - start with a loop that goes through all of the training data once._
  - _**Hint** - remember the inputs `train.x` and outputs `train.y` are stored separately (inputs & outputs for your batch) - make sure you use the same range for selecting both._
  - _**Hint** - if you're unsure if it is working, print out the labels, are they different each time?_


## 5. Evaluating the Model

We're very nearly there! If you've kept up so far, there are two outstanding issues (fortunately both are easy to fix):

 1. We are only reporting results for our training data - what about overfitting?
 2. Our results (cross entropy loss) are hard to interpret.

For the first, simply evaluate (but **do not take an optimization step**) on your validation set periodically.

_**Exercise** - run your network to report validation cross entropy loss every time you pass through the training data, without taking an optimization step on it (hint: you can do this in one big batch)._

_**Exercise** - use the `dlt.Log` class to plot validation & training curves._

To address the second issue, we can use `C.functions.accuracy` (which is run identically to `softmax_cross_entropy`) to report how many times our top prediction for the label is correct.

_**Exercise** - report accuracy for your network - before training it should be 1-2% - has it improved?_

_**Exercise** - try your classifier out by hand with the custom input control (N.B. remember you need a batch dimension `N`, even if `N=1`.)_


## 6. Iterate!

If you have followed the exercises so far, you should have a single layer `Linear` network, trained using `SGD`, using batch size `128`, passing `4` times over the training data. The accuracy probably isn't very good, so why not try (in any order):

 - Create a more complex function:
     - You started with `(N, 256) -> Linear(256, 62) -> (N, 62)`.
     - Maybe try `(N, 256) -> Linear(256, D) -> Tanh -> Linear(D, 62) -> (N, 62)` for some choice of `D` (this is a classic multilayer perceptron). See Appendix A1 for details on how to train this network (as you need to create a single link containing both linear links, to be able to train it).
     - Follow your imagination (see Appendix A2)...
 - Try another [optimizer](http://docs.chainer.org/en/stable/reference/optimizers.html) (e.g. Adam, AdaDelta, RMSprop).
 - Increase the number of passes through the training data (but try to stay `<= 10`, for sake of time).
 - Change the batch size.

_**Exercise** - improve the accuracy of your system - can you make it to 50%, 60%, 70%, 80%, 90% on the validation set?_


## A1. Deep learning using Chains

Chainer composes objects using the aggregate pattern. A `Chain` is a `Link` which can contain many of it's own component `Links`. If this sounds complex, don't worry! For us, all we'll do is make a single Chain that contains all the Links we use. For example, if we were running:

    x = ...                       # shape: (N, 3)
    first = C.links.Linear(3, 5)
    second = C.links.Linear(5, 7)
    a = first(x)                  # shape: (N, 5)
    y = second(a)                 # shape: (N, 7)

    # Problem - we can't call `optimizer.setup()` with both `first` and `second`!

Now we can combine the Links into a Chain:

    class Network(C.Chain):
        def __init__(self):
            super().__init__(
                first=C.links.Linear(3, 5),  # only links need to go here, not functions
                second=C.links.Linear(5, 7),
            )

        def __call__(self, x):
            '''x -- shape: (N, 3)
            returns -- shape: (N, 7)
            '''
            a = self.first(x)      # shape: (N, 5)
            return self.second(a)  # shape: (N, 7)

    network = Network()
    y = network(x)

    # Now we can use `optimizer.setup(network)` - problem solved!


## A2. Toolbox

To design a deep learning network, you'll use a bunch of standard components. To find out more, explore the Chainer [function](http://docs.chainer.org/en/stable/reference/functions.html) and [link](http://docs.chainer.org/en/stable/reference/links.html) docs, but here are some relevant ones:

| Link/Function | Transforms shape | Purpose |
| ------------- | ---------------- | ------- |
| C.links.Linear | `(N, A) -> (N, B)` | Changing dimension & general computation |
| C.functions.tanh | `(N, X) -> (N, X)` | Separating linear layers for more interesting computation |
| C.functions.sigmoid | `(N, X) -> (N, X)` | Like tanh, but sometimes used as a "gate", as outputs are [0 1] |
| C.functions.relu | `(N, X) -> (N, X)` | Another _activation function_ (like tanh, sigmoid) - there are loads of these, doing similar jobs! |
| C.links.Highway | `(N, X) -> (N, X)` | A fancy combination of identity & a square linear transform |
| C.functions.softmax\_cross\_entropy | `(N, X), (N,) -> ()` | Multiclass classification loss function, to compare a batch of scores against target labels |
| C.functions.accuracy | `(N, X), (N,) -> ()` | Multiclass classification accuracy, for evaluation |
