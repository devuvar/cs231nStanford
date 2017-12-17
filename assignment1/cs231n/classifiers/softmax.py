import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    #normalize so that max score is 0
    scores -= np.max(scores)
    nr = np.exp(scores[y[i]])
    dr = np.sum(np.exp(scores))
    l_i = -1 * np.log(nr/dr)
    #l_i = -scores[y[i]] +np.log(dr)
    loss += l_i
    
    #gradient computation
    scores_k = np.exp(scores)/dr
    for k in xrange(num_class):
        delta = 0
        if y[i]==k:
            delta =1
        dW[:,k] += (scores_k[k] - delta) * X[i]
    
  loss/=num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  #normalize so that max score is 0
  scores -= np.max(scores, axis=1, keepdims=True)

  scores_y = scores[range(num_train), y]

  exp_scores = np.exp(scores)
    
  nr = np.exp(scores_y)
  dr = np.sum(exp_scores, axis=1)

  #loss = -np.sum(np.log(nr/dr))
  loss = -np.sum(np.log(nr)-np.log(dr))
  #l_i = -scores[y[i]] +np.log(dr)
  #compute gradient
  p = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  ind = np.zeros_like(scores)
  ind[range(num_train),  y] = 1
  dW = X.T.dot(p - ind)

  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

