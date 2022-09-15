# Simple Models

We have arrived at the good part: actual machine learning! Well, not exactly… but we are close. To understand the more complex algorithms, it is important to first understand the basics. Do not be discouraged, because “simple” models oftentimes outperform very complex models. Not only are they more interpretable, but also easier to train. For most of the models described in this part, we do not need fancy GPUs and big supercomputers to train everything, but we can do it within seconds (depending on the size of the dataset, of course). Still, some of the stuff is quite difficult to understand – especially if you have trouble with math. I will try, however, to explain these methods as simple as possible, without losing too much information.

## In this file

- [Linear Regression](#linear-regression)
- [Logistic Regression](#logistic-regression)
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

![afbeelding](https://user-images.githubusercontent.com/89044870/187948667-67e8afb9-c6ca-40a3-b96f-0aab26e1f8b8.png)

It is obvious that we cannot come up with a linear function that describes all the data points exactly. Therefore, we want to approximate a linear function that describes the relationship between the x and the y axis as accurately as possible.

To do so, we use the least squares method. Without boring you with all of the details, it is most important to know the following: it is an approximation method to minimize the sum of squared residuals. As explained in the losses section, a residual is a difference between the observed value and the estimated value of the quantity of interest (e.g. sample mean). Using a lot of math, it can be rewritten as the following formula:

$\hat{\beta} = (X^TX)^{-1}X^TY$

Note that $\hat{\beta}$ is the estimated coefficient matrix. The implementation of this in Python looks complicated but is actually very straightforward. Take a look at the snippet below:

![afbeelding](https://user-images.githubusercontent.com/89044870/187948748-7e06fec2-1813-4c35-a6f0-74ab91cb09a9.png)

And we are done! Yes, this is everything that is needed to implement linear regression from scratch. If we run our linear regression model on the datapoints from before, we get the following linear estimation:

![afbeelding](https://user-images.githubusercontent.com/89044870/187948802-7072edb6-6fd5-4c05-852e-187e0cf07a2a.png)

Which looks very good. However, to estimate the goodness of a fit, the R-squared measure is most often used for linear regression. If we use our R-squared loss function, we find a score of 0.842. Remember, the closer to 1, the more variance in the data is explained by the model. Given that our model has quite a bit of noise, I would say that an 84.2% explanation of variance is a very good result.

### When to use it

Linear regression works best when there is clear linearity in your data. It is, however, not always easy to visualize this (e.g. when the dimensionality becomes too big). Nonetheless, when facing a regression problem it is often useful to start with linear regression to test for linearity and discover how well this computationally efficient, simple, explainable method can fit your data.

***Example use cases***

- Predicting how the change in the price of product x will impact the sales of that product.
- Determining the crop yield given the amount of fertilizer and water that you use.
- Modeling the relationship between income and happiness.
- Many more…

## Logistic Regression

I hope you paid a lot of attention to the explanation of linear regression since logistic regression is very similar to it. Whereas linear regression is used to measure the relationship between a **scalar** response and one or multiple explanatory variables, logistic regression is used to measure the **probability** of an event given one or multiple explanatory variables. Another key difference is that this probability is not linear, but uses the logistic function to describe the probability of an event happening (more on that later).

Let’s look at an example. Say we want to describe the relationship between hours of studying and the probability of passing a class. In my experience – and I assume many others share that experience – if I spend more hours studying, I have a higher chance of passing a class. However, studying for 1 hour instead of not at all does not improve my grasp of the material (and therefore my chances of passing the class) as much as, say, studying for 3 hours instead of 2. Conversely, there is also a limit to when studying for longer only improves my chances marginally. The relationship is thus not linear but can be described using the logistic function.

As you can imagine, this probability can be used to predict a (binary) class. In the case of the previous example, we might say that if a student has a probability of passing the class that is above 50% will be classified as “will pass the class”, whereas anything below that probability will be classified as “will fail the class”.

Now, onto the math. You may recall that for linear regression, we tried to estimate the coefficients (weights vector) for the function $y=\textbf{w}^T\textbf{x}$. Logistic regression still uses the same concept of trying to estimate the coefficients, but also takes the logistic function into account, since we are trying to predict a class and not a real number. So, we have to optimize for the probability of the class. The logistic function can be described as follows:

$p(x) = \frac{1}{1 + e^{-(\textbf{w}^T\textbf{x})}}$

As you can see, this function uses the aforementioned linear function ($y=\textbf{w}^T\textbf{x}$) to estimate the probability of this datapoint labeled as class 1 by scaling it from 0 to 1.

### Implementation

The most straightforward method of solving a logistic progression problem is by approximating the coefficients using “gradient descent”. This is an optimization algorithm that tries to find a local minimum of an error or loss function. It should be noted that the weights for linear regression can also be estimated using this method. The most common error function used in binary classification is the binary cross-entropy function. For this implementation, it is not necessary to know the specifics, but just how to calculate the partial derivatives with regards to the bias and the weights (i.e. the gradient).

$\partial \textbf{w} = \frac{1}{m}(\hat{\textbf{y}}-\textbf{y})\textbf{x}^T$

$\partial \textbf{w} = \frac{1}{m}\sum(\hat{\textbf{y}}-\textbf{y})$

Where *m* denotes the number of elements in ***x***.

Now that we know how to calculate the partial derivatives of the binary cross-entropy function with regard to the weights and bias, we can update our weights and bias accordingly. To do so, we first need to set a *learning rate*. This means that we have to determine how much we want to change our weights each epoch. If we set the learning rate too high, we can get undesirable divergent behavior, whereas if the learning rate is too low, training the model will take much longer than necessary. The learning rate is usually a value between 0.1 and 0.0001. This is not a value that you can know beforehand but has to be estimated using trial-and-error.

After having set the learning rate, we have everything we need to build our logistic regressor. We start by initializing our weights and bias to zeros. Then, we need to first calculate the probabilities of each datapoint in our dataset using the sigmoid function. Next, we calculate the partial derivatives and update our weights using the learning rate we set.
We can implement this in Python as follows:

![afbeelding](https://user-images.githubusercontent.com/89044870/190502903-95d58415-7a30-4b54-bc4d-55c67931f9ba.png)

That is it! Now, the only thing that rests us, is the prediction. To predict a datapoint, we simply calculate the sigmoid for that datapoint again, and if that value is above a certain threshold, we classify the datapoint as 1. Otherwise, we classify it as 0.

### When to use it

As stated before, this implementation of logistic regression can only be used for binary classification problems. Again, it is a very useful, simple, and explainable method that can be used as a starting point for many classification problems. It might surprise you how well this simple model relates to many (seemingly difficult) classification problems.

***Example use cases***

- Predicting how many hours I need to study to pass my class.
- Predicting the likelihood of a disease given the patient's symptoms.
- Many other binary classification problems…

## Decision Trees

TBA
