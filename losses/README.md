# Loss functions

**Error** or **loss functions** are some of the most important things to consider when training and testing your model. It is the way we tell our model how well it is performing. If the model is performing poorly, we must let it know that it is doing so, so it knows that it has some area to improve upon and it needs to rearrange some weights (i.e., reweigh information differently to get to a better answer). Different modelling problems require different loss functions. This section aims to describe the implementation and use of some of these functions.

## In this file

* [Numerical losses](#numerical-losses)
  * [R-Squared](#r-squared)
  * [Mean Squared Error (MSE)](#mean-squared-error-mse)
  * [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
* [Categorical losses](#categorical-losses)

## Numerical losses

### R-Squared

The R-Squared or R^2 loss function is mostly used in Linear Regression and is a measure of describing how well the regression predicts the approximate real data points. It describes how much the variance in the dependent variable is explained by the independent variables collectively. In other words: how well your model can explain the difference between the ground truth and its prediction. If you have a higher R-Squared, your model better fits your observations. If you have a lower R-Squared, your model cannot explain all the variance in the response variable. This measure is scaled from 0-100%. There is no need to panic if your calculated R^2 is lower than 100%, since there might not exist a perfect fit for this data using the model type you use. It might not be possible to create a perfect fit for your model because of noise in your data, or maybe you should thing about using another type of model.
Because we want to know the variability of data around the mean, we can calculate the R-Squared using two sum of squares formulas. First, we calculate the sum of squares of residuals (residual sum of squares). The residual is the difference between the observed value and the estimated value of the quantity of interest (e.g., sample mean). Then, we also want to calculate the total sum of squares. Once we have those, we can calculate our loss.

Sum of squares of residuals: $SS_{res} = \sum_i(y^i_{true} - y^i_{pred})^2$

Sum of total variance: $SS_{tot} = \sum_i(y^i_{true} - mean(y_{true}))^2$

R-squared: $R^2 = 1-\frac{SS_{res}}{SS_{tot}}$

Using these formulas, the implementation in Python is very straight-forward:
![afbeelding](https://user-images.githubusercontent.com/89044870/187924600-e4cc15c5-b544-4ca4-91aa-c24731731f18.png)

### Mean Squared Error (MSE)

The mean squared error (MSE) is a very straight-forward loss function: for each of the datapoints, we calculate the difference between the prediction and the ground truth. To punish big errors more than small errors, we square it. Then, we want to average these squared errors to get the intuitive mean squared error. This error is scaled from 0 to infinity. It is used for all sorts of problems but is most often employed when we want to penalize big differences. When predicting crowdedness for the metro for example, we do not really mind if we predict 3 passengers, and there are truly 4. However, we do not want to predict 300 passengers, when there are 400. This could also be achieved with the mean absolute error, but because that loss is linear, there is not as much “urgency” to the model to correct these predictions.

The implementation in Python is also very simple, and can be described in one line of code:
![afbeelding](https://user-images.githubusercontent.com/89044870/187924694-29fea803-b8d5-4b38-a688-597d694da407.png)

### Mean Absolute Error (MAE)

The mean absolute error (MAE) is even more straight-forward and easy to understand than the MSE. In essence: we sum the absolute value of the prediction, subtracted by the true value for each of our datapoints and take its mean (i.e. divide it by N datapoints).
This is a very general-purpose loss-function, and is used when big errors are not that important. Another benefit of this error is that it is very intuitive and can therefore be explained to the layman very easily. It is simply the average "mistake".

This implementation in Python is similar to the MSE, and can also be described in one line of code:
![afbeelding](https://user-images.githubusercontent.com/89044870/187924736-dae68d37-d732-48a8-9d2f-37510244cc1e.png)

## Categorical losses

TBA
