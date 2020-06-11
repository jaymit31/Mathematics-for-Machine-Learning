Mean/Covariance of a data set and effect of a linear transformation
We are going to investigate how the mean and (co)variance of a dataset changes when we apply affine transformation to the dataset.

Learning objectives
Get Farmiliar with basic programming using Python and Numpy/Scipy.
Learn to appreciate implementing functions to compute statistics of dataset in vectorized way.
Understand the effects of affine transformations on a dataset.
Understand the importance of testing in programming for machine learning.
First, let's import the packages that we will use for the week

# PACKAGE: DO NOT EDIT THIS CELL
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import time
import timeit
%matplotlib inline
from ipywidgets import interact
Next, we are going to retrieve Olivetti faces dataset.

When working with some datasets, before digging into further analysis, it is almost always useful to do a few things to understand your dataset. First of all, answer the following set of questions:

What is the size of your dataset?
What is the dimensionality of your data?
The dataset we have are usually stored as 2D matrices, then it would be really important to know which dimension represents the dimension of the dataset, and which represents the data points in the dataset.

When you implement the functions for your assignment, make sure you read the docstring for what each dimension of your inputs represents the data points, and which represents the dimensions of the dataset!. For this assignment, our data is organized as (D,N), where D is the dimensionality of the samples and N is the number of samples.

image_shape = (64, 64)
# Load faces data
dataset = fetch_olivetti_faces('./')
faces = dataset.data.T
​
print('Shape of the faces dataset: {}'.format(faces.shape))
print('{} data points'.format(faces.shape[1]))
Shape of the faces dataset: (4096, 400)
400 data points
When your dataset are images, it's a really good idea to see what they look like.

One very convenient tool in Jupyter is the interact widget, which we use to visualize the images (faces). For more information on how to use interact, have a look at the documentation here.

We have created two function which help you visuzlie the faces dataset. You do not need to modify them.

def show_face(face):
    plt.figure()
    plt.imshow(face.reshape((64, 64)), cmap='gray')
    plt.show()
@interact(n=(0, faces.shape[1]-1))
def display_faces(n=0):
    plt.figure()
    plt.imshow(faces[:,n].reshape((64, 64)), cmap='gray')
    plt.show()
interactive(children=(IntSlider(value=0, description='n', max=399), Output()), _dom_classes=('widget-interact'…
1. Mean and Covariance of a Dataset
In this week, you will need to implement functions in the cell below which compute the mean and covariance of a dataset.

You will implement both mean and covariance in two different ways. First, we will implement them using Python's for loops to iterate over the entire dataset. Later, you will learn to take advantage of Numpy and use its library routines. In the end, we will compare the speed differences between the different approaches.

# GRADED FUNCTION: DO NOT EDIT THIS LINE
def mean_naive(X):
    "Compute the mean for a dataset X nby iterating over the data points"
    # X is of size (D,N) where D is the dimensionality and N the number of data points
    D, N = X.shape
    mean = np.zeros((D, 1))
    mean = (np.sum(X,axis=1)/N).reshape(D,1)
    return mean

def cov_naive(X):
    """Compute the covariance for a dataset of size (D,N)
    where D is the dimension and N is the number of data points"""
    D, N = X.shape
    covariance = np.zeros((D, D))
    temp = X - mean_naive(X)
    covariance = (temp @ temp.T)/N
    return covariance
​
def mean(X):
    "Compute the mean for a dataset of size (D,N) where D is the dimension and N is the number of data points"
    # given a dataset of size (D, N), the mean should be an array of size (D,1)
    # you can use np.mean, but pay close attention to the shape of the mean vector you are returning.
    D, N = X.shape
    ### Edit the code to compute a (D,1) array `mean` for the mean of dataset.
    mean = np.zeros((D,1))
    mean = np.mean(X, axis=1, keepdims=True)
    ###
    return mean
​
def cov(X):
    "Compute the covariance for a dataset"
    # X is of size (D,N)
    # It is possible to vectorize our code for computing the covariance with matrix multiplications,
    # i.e., we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    # We challenge you to give a vectorized implementation without using np.cov, but if you choose to use np.cov,
    # be sure to pass in bias=True.
    D, N = X.shape
    ### Edit the code to compute the covariance matrix
    covariance_matrix = np.zeros((D, D))
    ### Update covariance_matrix here
    covariance_matrix = cov_naive(X)
    ###
    return covariance_matrix
Now, let's see whether our implementations are consistent

# Let's first test the functions on some hand-crafted dataset.
​
X_test = np.arange(6).reshape(2,3)
expected_test_mean = np.array([1., 4.]).reshape(-1, 1)
expected_test_cov = np.array([[2/3., 2/3.], [2/3.,2/3.]])
print('X:\n', X_test)
print('Expected mean:\n', expected_test_mean)
print('Expected covariance:\n', expected_test_cov)
​
np.testing.assert_almost_equal(mean(X_test), expected_test_mean)
np.testing.assert_almost_equal(mean_naive(X_test), expected_test_mean)
​
np.testing.assert_almost_equal(cov(X_test), expected_test_cov)
np.testing.assert_almost_equal(cov_naive(X_test), expected_test_cov)
X:
 [[0 1 2]
 [3 4 5]]
Expected mean:
 [[ 1.]
 [ 4.]]
Expected covariance:
 [[ 0.66666667  0.66666667]
 [ 0.66666667  0.66666667]]
We now test that both implementation should give identical results running on the faces dataset.

np.testing.assert_almost_equal(mean(faces), mean_naive(faces), decimal=6)
np.testing.assert_almost_equal(cov(faces), cov_naive(faces))
With the mean function implemented, let's take a look at the mean face of our dataset!

def mean_face(faces):
    return faces.mean(axis=1).reshape((64, 64))
​
plt.imshow(mean_face(faces), cmap='gray');

Loops in Python are slow, and most of the time you want to utilise the fast native code provided by Numpy without explicitly using for loops. To put things into perspective, we can benchmark the two different implementation with the %time function in the following way:

# We have some HUUUGE data matrix which we want to compute its mean
X = np.random.randn(20, 1000)
# Benchmarking time for computing mean
%time mean_naive(X)
%time mean(X)
pass
CPU times: user 48 µs, sys: 23 µs, total: 71 µs
Wall time: 74.9 µs
CPU times: user 90 µs, sys: 42 µs, total: 132 µs
Wall time: 141 µs
# Benchmarking time for computing covariance
%time cov_naive(X)
%time cov(X)
pass
CPU times: user 475 µs, sys: 223 µs, total: 698 µs
Wall time: 714 µs
CPU times: user 233 µs, sys: 0 ns, total: 233 µs
Wall time: 242 µs
As you can see, using Numpy's functions makes the code much faster! Therefore, whenever you can use something that's implemented in Numpy, be sure that you take advantage of that.

2. Affine Transformation of Datasets
In this week we are also going to verify a few properties about the mean and covariance of affine transformation of random variables.

Consider a data matrix $\boldsymbol X$ of size $(D, N)$. We would like to know what is the covariance when we apply affine transformation $\boldsymbol A\boldsymbol x_i + \boldsymbol b$ for each datapoint $\boldsymbol x_i$ in $\boldsymbol X$, i.e., we would like to know what happens to the mean and covariance for the new dataset if we apply affine transformation.

For this assignment, you will need to implement the affine_mean and affine_covariance in the cell below.

# GRADED FUNCTION: DO NOT EDIT THIS LINE
def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        x: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    ### Edit the code below to compute the mean vector after affine transformation
    affine_m = np.zeros(mean.shape)
    affine_m = A @ mean + b
    ### Update affine_m

    ###
    return affine_m
​
def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation
    Args:
        S: ndarray, the covariance matrix
        A, b: affine transformation applied to each element in X
    Returns:
        covariance matrix after the transformation
    """
    ### EDIT the code below to compute the covariance matrix after affine transformation
    affine_cov = np.zeros(S.shape)
    affine_cov = A @ S @ A.T
    ### Update affine_cov

    ###
    return affine_cov
Once the two functions above are implemented, we can verify the correctness our implementation. Assuming that we have some $\boldsymbol A$ and $\boldsymbol b$.

random = np.random.RandomState(42)
A = random.randn(4,4)
b = random.randn(4,1)
Next we can generate some random matrix $\boldsymbol X$.

X = random.randn(4,100) # D = 4, N = 100
Assuming that for some dataset $\boldsymbol X$, the mean and covariance are $\boldsymbol m$, $\boldsymbol S$, and for the new dataset after affine transformation $\boldsymbol X'$, the mean and covariance are $\boldsymbol m'$ and $\boldsymbol S'$, then we would have the following identity:

$$\boldsymbol m' = \text{affine_mean}(\boldsymbol m, \boldsymbol A, \boldsymbol b)$$

$$\boldsymbol S' = \text{affine_covariance}(\boldsymbol S, \boldsymbol A, \boldsymbol b)$$

X1 = (A @ X) + b  # applying affine transformation to each sample in X
X2 = (A @ X1) + b # twice
One very useful way to compare whether arrays are equal/similar is use the helper functions in numpy.testing.

Check the Numpy documentation for details. The mostly used function is np.testing.assert_almost_equal, which raises AssertionError if the two arrays are not almost equal.

np.testing.assert_almost_equal(mean(X1), affine_mean(mean(X), A, b))
np.testing.assert_almost_equal(cov(X1),  affine_covariance(cov(X), A, b))
np.testing.assert_almost_equal(mean(X2), affine_mean(mean(X1), A, b))
np.testing.assert_almost_equal(cov(X2),  affine_covariance(cov(X1), A, b))
