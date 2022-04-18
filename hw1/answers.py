r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. **False**, the in sample error is the error rate you get on the same data set you used to build your predictor.  
The test set is not used to build the predictor, hence we cannot use it to estimate the in-sample error.  
The test set can help us estimate the performance on the data that we have, not on all possible data.  
The distribution of the test set data may be different that the training data.

2. **False**, we want the distribution of both parts to resemble the real-world data distribution.  
Some of the splits may be distributed differently. For example, if we classify between two colors, it is not equally
useful to choose a training set with one color and a test set with the other.  
In addition, we want the training set to be much larger than the test set to avoid overfitting.  
A common practice is to use only 10-20% of the data for the test set.  

3. **False**, cross validation is a part of the training process, in which the test set should not be used at all.  
If we will use the test set in the process, we may overfit the data so we get a very low error rate on the test set,
but the classifier will not perform correctly on real world data.  

4. **True**, we use manifold cross validation to estimate the generalization error on real world data. Since we did not
learn on the data of the validation set, this data can help us estimate the performance of our model on real world data.
After attempting different test sets, the one with the lowest estimation error will give an evaluation of the model's 
performance.

"""

part1_q2 = r"""
The approach **is justified**. The regularization parameter $\lambda$ reduces overfitting, which reduces the variance of
 the estimated parameters $w$ of the hyperplane, at the expense of adding bias to their estimation. Increasing $\lambda$
 results in less overfitting, but also greater bias. E.g minimizing the cost function will come at the expense of the
 results of the regression model on the training set. A way of finding out if the bias is too great is to look at the
 results fo the test set.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
Increasing k **does not** necessarily leads to improved generalization for unseen data. This is dependant on the
 distribution of the data. Incrasing K may improve the generalization up to a point where most of the nearest neighbors
 do not belong to the same class. The extreme case is where k equals the size of the training data, and the
 classification result will always equal the majority class in the training data.
"""

part2_q2 = r"""
1. Training set accuracies does not tell us the generalization on unseen data. E.g when choosing the best model
 according to train set accuracies, we actually choose the most overfitted model. k-fold CV tests the model
 accuracy on unseen data, which gives us a more generalized model.
2. The purpose of the test set is to estimate the model's performance, using it as a "validation set" will cause us
 to choose a model which is hypertuned to the test set, but it will not necessarily perform as well on a similarly
 distributed real world data set.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
