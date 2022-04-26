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
The $\Delta$ parameter represents the initial margin in the soft SVM model. The difference between soft SVM and hard 
 svm is that we allow sample classification to be inside the margin, and the $\Delta$ variable represents the penalty
 we give for solutions inside the margin. During the training process, the margins update, and their distance change by
 the regularization term $\lambda  ||W||^2$. The weights will be updated to minimize the penalty.
 E.g by hypertuning $\lambda$ and updating the margins, the $\Delta$ variable becomes irrelevant during the 
 trainig process.
 """

# outdated - this describes an example of the results with low learning rate.
#            in better results the `1`s are not misclassified.
# '''
#  For example, we can see that the learnt template for the `1` class is indented to the right. In one of the examples in
#  the test set, the image of `1` is indented to the left and the sample is misclassified as `3`. This can hint that the
#  training set did not contain enough samples of `1` which are indented to the left, e.g the distribution of this class
#  in the training set is biased towards right-indented `1`s.
# '''

part3_q2 = r"""
1. We can essentially see that for each class, we learn a `template` which has the same size as the input images, and 
 has positive and negative values for each pixel. The score we get for each class is by template matching between the
 template and the input image. If the input image has a positive value at a pixel and the template has a positive value
 at the same pixel, it will contribute to the likelihood that the image belongs to the class. If the template has a
 negative value at this pixel, it will reduce the likelihood. If the template at this pixel is close to zero, than this
 pixel has almost no contribution to the result.  
 In essence, the learnt template defined areas in the images that help us to classify them into the classes.  
 For example, we can see that there are a lot of mis-classification in the `6` class. When we look at the learnt
 weights for this class, we see that it does not resemble `6`. 


2. KNN does not have ha training process. Instead it compares the sample with all of the data it has, which is time
 consuming, and the only parameter we can change is the number of neighbors to check.  
 The SVM model has a compact representation of the statistics of each class in its weights, reducing inference time
 compared to KNN. The similarity between the two models is that in both cases we look at pixel values as features and
 we do not have any other feature extraction mechanism (both use the data as a vector of 28x28 pixels).

"""

part3_q3 = r"""
* Learning Rate - The rate is good. When it was too hight, the trainig process did not converge and we didn't get
 acceptable results. When it was too low, too many epochs have passed and we did not get results as good as with
 the learning rate we chose (local minimum which is higher than the one we got).

* Overfitting - We can see that the model is slightly overfitted. From about epoch #8 and beyond, we can see that the 
 loss of the validation set becomes slightly larger than of the training set and that the accuracy on the training set
 is slightly better than on the validation set.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern to see in the residual plot is that all of the points are close to the line (small
 estimation error). The points of the training set (blue) should not be less spread than the points of the test set 
 (orange), since this could hint at overfitting.  
 Looking at the residual plot of the final model, we can see that although the mse of the test set is larger
 than the mse of the training set, the spread of estimation errors is similar for both sets.
 Comparing this plot with the plot of the best five features, we can see that for each of these individual features,
 the spread of the data points around the fitted line is large, which gives us an indication that none of this features
 can explain alone the behaviour of the MEDV variable. The final plot is not a plot of one "explained" variable
 vs. one "explaining" variable, but of the estimation error vs the variable. These are two different plots, but the
 clustering of data points closer to the line suggests that the estimator "explains" the MEDV variable much better
 than any single feature.  

"""

part4_q2 = r"""
1. **This is still a linear regression model**.  
 In a linear regression model, estimate the parameters of a linear function between
 between the data and the estimated variable. What we did is to fit a linear function to some "features" data.
  
2. Theoretically, we **can** fit any non-linear function of the feauters with this approach.
   This can happen only if we have some intuition about the non-linearity of the function. Using this assumption,
   we can create "building blocks", e.g. non linear functions of the data and linearly fit those blocks to
   the data.
   
3. The decision boundary will **still be a hyperplane**. Using non-linear features, we cast the original data to
 another dimension space, and the hyperplane is the decision boundary in that space. For example, consider the
 the function $y=x_1^2 + x_2^2$. If we want to distinguish between points at the threshold of y<3 and y>=3, we cannot
 do this as a linear function of $x_1$ and $x_2$. However, we can do this using a linear function of $x_1^2$ and $x_2^2$ 
"""

part4_q3 = r"""
1. The `logspace` was used instead of `linspace` because the `logspace` defines a list of numbers in different orders of 
 magnitudes, where `linspace` defines a list of numbers in the same order of magnitude. The $\lambda$ parameter gives 
 weight to the magnitude of the parameters we want to estimate. By choosing a very small value **compared to the cost
 without the regularization term**, the total cost will not change by much if the magnitude of the estimated parameters
 will be large, which can increase the chance of overfitting.  
 In contrast, by choosing a $\lambda$ which is orders of magnitude larger than the cost function without
 the regularization term, changes in the parameters will not change the total cost by much, which will make the
 convergence of the process very hard. The order of magnitude of $\lambda$ should be such that the order of magnitude of
 the regularization term is the same as the order of magnitude of the cost function without the regularization term.  
 By choosing `logspace` instead of `linspace`, we test different orders of magnitude.
2. With CV, the number of times the model was fit it $3_{[degree options]} * 20_{[\lambda options]} * 3_{[folds]} = 180$
"""

# ==============
