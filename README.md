# TiML

Hi there! Welcome to my small project. In this repository I have recreated some (minimally functioning) implementations of some of the most commonly used machine learning methods.

## Why this project?

I know that I am basically reinventing the wheel with a lot of these implementations, but I have started this repository as an exercise to myself, and as a help to new machine learning enthousiasts. Even though I have a bachelor's degree in Artificial Intelligence, and am a second year AI master's student, I also sometimes still struggle with the abundance of techniques available, and the options to choose from when starting a new project. For any project, one can opt for not only different kinds of models, but additional choices need to be made along with the careful consideration of each model. Choosing the correct optimizer, learning rate, loss function, train/test split size are just a few of the general considerations. Then choices need to be made about model-specific optimizations. To me and some of my fellow peers, these choices are just too many, and it is impossible to understand each one of them. However, to effectively use these powerful methods, it is important to – at the least – have a vague notion of what they mean.

## About this repository

With this in mind, I have attempted to document the implementation of various machine learning algorithms, along with explanations, information, and some illustrations. Most Python implementations require the use of the NumPy library. It is highly recommended that you install this library as well, if you want to replicate any of my work. In this repository, I have also included a `.pdf` file that contains more all the combined information that I have collected and written down for this project, along with some of my thoughts. The actual code in this document might differ slightly from some of the code snippets included in this repo. The code, however, does not work any differently. It has purely been made easier to read for the layman here.
You will find that in each of the subfolders, there is also a corresponding `README.md` file that contains (excerpts of) the relevant parts in the `.pdf` file.

Please be advised that I am currently in the early stages of this project and the code in the repo, along with the given information probably contain many mistakes. If you stumble upon such mistakes or have any other questions, please, do not hesitate to send an email to tim.koornstra@gmail.com or create an “issue” on the GitHub page for this project.

## Future methods
- Decision trees
- Logistic regression
- Support vector machine
- Single layer perceptron
- Multi layer perceptron
- K-nearest neighbors
- K-means clustering

## Dependencies

Currently, this project's only dependency is `NumPy`. It is strongly advised to install this library as well. This can be done as follows:

```bash
pip install numpy
```

## How to run

1. Clone the repository
2. Create a `main.py` file in the root of this repo
3. In this file, import any files that you wish to use. To import, use:

    ```py
    from folder.file import *
    ```

4. Run your `main.py` file using your preprocessed data and the imported libraries
