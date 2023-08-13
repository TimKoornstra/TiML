# TiML

Hi there! Welcome to my small project. In this repository I have recreated some (minimally functioning) implementations of some of the most commonly used machine learning methods.

## Why this project?

I know that I am basically reinventing the wheel with a lot of these implementations, but I have started this repository as an exercise to myself, and as a help to new machine learning enthousiasts. Even though I have a bachelor's degree in Artificial Intelligence, and am a second year AI master's student, I also sometimes still struggle with the abundance of techniques available, and the options to choose from when starting a new project. For any project, one can opt for not only different kinds of models, but additional choices need to be made along with the careful consideration of each model. Choosing the correct optimizer, learning rate, loss function, train/test split size are just a few of the general considerations. Then choices need to be made about model-specific optimizations. To me and some of my fellow peers, these choices are just too many, and it is impossible to understand each one of them. However, to effectively use these powerful methods, it is important to – at the least – have a vague notion of what they mean.

## About this repository

With this in mind, I have attempted to document the implementation of various machine learning algorithms, along with explanations, information, and some illustrations. Most Python implementations require the use of the NumPy library. It is highly recommended that you install this library as well, if you want to replicate any of my work. In this repository, I have included an "examples" folder where you can find more elaborate explanations and examples of the code. The actual code in those notebooks might differ slightly from some of the code snippets included in this repo. The code, however, does not work any differently. It has purely been made easier to read for the layman here.

Please be advised that I am currently in the early stages of this project and the code in the repo, along with the given information probably contain many mistakes. If you stumble upon such mistakes or have any other questions, please, do not hesitate to send an email to tim.koornstra@gmail.com or create an “issue” on the GitHub page for this project.

## Methods in this repository
| Method                  | Programming done? | Example notebook done? |
|-------------------------|-------------------|---------------------|
| Linear Regression       | ✅                 | ✅                   |
| Decision Tree           | ✅                 | ❌                   |
| Logistic Regression     | ✅                 | ❌                   |
| Support Vector Machine  | ❌                 | ❌                   |
| Single Layer Perceptron | ❌                 | ❌                   |
| Multi Layer Perceptron  | ❌                 | ❌                   |
| K-Nearest Neighbors     | ❌                 | ❌                   |
| K-Means Clustering      | ❌                 | ❌                   |
| Naive Bayes Classifier  | ❌                 | ❌                   |

## Dependencies

If you would like to build the project from source, you would need to install some requirements.

Currently, this project's only dependency is `NumPy`. It is strongly advised to install this library as well. This can be done as follows:

```bash
pip install numpy
```

Or - alternatively - installing all requirements at once can be done like so:

```bash
pip install -r requirements.txt
```

## How to run

1. Clone the repository
2. Install the library using pip by running
```bash
pip install .
```
