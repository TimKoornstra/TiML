# Simple Models

We have arrived at the good part: actual machine learning! Well, not exactly… but we are close. To understand the more complex algorithms, it is important to first understand the basics. Do not be discouraged, because “simple” models oftentimes outperform very complex models. Not only are they more interpretable, but also easier to train. For most of the models described in this part, we do not need fancy GPUs and big supercomputers to train everything, but we can do it within seconds (depending on the size of the dataset, of course). Still, some of the stuff is quite difficult to understand – especially if you have trouble with math. I will try, however, to explain these methods as simple as possible, without losing too much information.

## In this file

- [Linear Regression](#linear-regression)
- [Decision Trees](#decision-trees)

## Linear Regression

Linear regression is not a method that is unique to the field of Artificial Intelligence. It is a well-known statistical approach for modeling the relationship between a scalar response and one or multiple explanatory variables. We can, for example, ask ourselves the question “How much will the value of my house increase if I were to repaint it?”. In this case, we want to know the price difference (the scalar response) when we repaint the house (an explanatory variable). Simple linear regression (i.e. linear regression with one response, and one variable) is best described by the following function:

$y = mx + b$

You might recognize this formula from high school math. In the previous example: *y* is our house price, *m* is the added value of one *x* amount of paint added to the house, and b is the original house price. We can imagine that if we do not paint our house (*x=0*), the price will remain the same as before, whereas the price should increase the more we paint our house (until it is fully painted).

Then there is also something called *multiple* linear regression. The amount of responses stays the same, but the amount of *variables* is now bigger than 1. We must rewrite our original formula to include not only multiple variables, but also multiple coefficients that belong to those variables. We get something that looks like the following:

$y=\sum^i (m_i x_i)+b$

Or, when using matrix notation:

$y = \textbf{w}^T\textbf{x}$

Notice that the y-intercept (the bias term *b*) has disappeared. This is because it is usually included within the weight and variable vectors.

Lastly, there is also *multivariate* linear regression, where we predict multiple correlated variables, rather than a single scalar variable. In matrix format, we usually get a function like the following:

$\textbf{y} = \textbf{w}^T\textbf{x}$

This looks very similar to the previous formula but has a *y* vector as output, rather than a *y* scalar.

### Implementation

To describe the implementation as best as possible, it is useful to add an illustrated example. We have the following data points in a scatter plot:
blabla.png

It is obvious that we cannot come up with a linear function that describes all the data points exactly. Therefore, we want to approximate a linear function that describes the relationship between the x and the y axis as accurately as possible.

To do so, we use the least squares method. Without boring you with all of the details, it is most important to know the following: it is an approximation method to minimize the sum of squared residuals. As explained in the losses section, a residual is a difference between the observed value and the estimated value of the quantity of interest (e.g. sample mean). Using a lot of math, it can be rewritten as the following formula:

$\hat{\beta} = (X^TX)^{-1}X^TY$

Note that $\hat{\beta}$ is the estimated coefficient matrix. The implementation of this in Python looks complicated but is actually very straightforward. Take a look at the snippet below:
snippet.png

And we are done! Yes, this is everything that is needed to implement linear regression from scratch. If we run our linear regression model on the datapoints from before, we get the following linear estimation:
results.png

Which looks very good. However, to estimate the goodness of a fit, the R-squared measure is most often used for linear regression. If we use our R-squared loss function, we find a score of 0.842. Remember, the closer to 1, the more variance in the data is explained by the model. Given that our model has quite a bit of noise, I would say that an 84.2% explanation of variance is a very good result.

## Decision Trees

TBA
