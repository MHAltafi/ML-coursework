# Machine Learning — University of Tehran

This repository contains my coursework and implementations for the **Machine Learning** course at **University of Tehran (Semester 4041)**.

The assignments cover a range of machine learning concepts, from probabilistic classification and statistical learning to feature selection, neural networks, and unsupervised learning with Gaussian Mixture Models. **This repository focuses exclusively on the coding questions of each assignment**, with each assignment's Jupyter Notebook containing the corresponding implementations and experiments.

Each assignment folder also contains a dedicated `README.md` with a more detailed description of its implementation, experiments, results, and key findings.

## Table of Contents

- [Assignments](#assignments)
  - [Assignment 1 — Probabilistic Classification](#assignment-1--probabilistic-classification)
  - [Assignment 2 — Statistical Learning and Nonlinear Regression](#assignment-2--statistical-learning-and-nonlinear-regression)
  - [Assignment 3 — Feature Selection and Neural Networks](#assignment-3--feature-selection-and-neural-networks)
  - [Assignment 4 — Expectation-Maximization and Gaussian Mixture Models](#assignment-4--expectation-maximization-and-gaussian-mixture-models)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Topics Covered](#topics-covered)

## Assignments

### Assignment 1 — Probabilistic Classification

The first assignment focuses on **probabilistic classification**, using Gaussian Discriminant Analysis and Naive Bayes classifiers.

The implementation includes:

- Gaussian Discriminant Analysis (GDA) with shared and class-specific covariance matrices.
- Dimensionality reduction using Principal Component Analysis (PCA).
- Three-class classification and decision boundary visualization.
- Classification with a rejection option.
- Gaussian Naive Bayes implemented from scratch.
- Bag-of-Words text classification using Naive Bayes.
- Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation.
- Dirichlet/Laplace smoothing and evaluation under different hyperparameter settings.
- Confusion matrix and classification accuracy analysis.

**Notebook:** [Assignment1.ipynb](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment1/Assignment1.ipynb)

**README:** [Assignment 1 README](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment1/README.md)

### Assignment 2 — Statistical Learning and Nonlinear Regression

The second assignment explores **statistical learning, nonparametric density estimation, statistical inference, and nonlinear regression**.

The implementation includes:

- Parzen Window density estimation for uniform and exponential distributions.
- Analysis of bandwidth selection and its effect on density estimation.
- Parzen Window classification using a spherical Gaussian kernel.
- Maximum Likelihood Estimation (MLE) for the normal distribution.
- Numerical and closed-form parameter estimation.
- Asymptotic covariance analysis and confidence interval construction.
- Likelihood Ratio, Wald, and Z hypothesis tests.
- Cross-validation and the Delta Method.
- Nonlinear regression using Gaussian basis functions.
- Ridge regularization and bias-variance tradeoff analysis.

**Notebook:** [Assignment2.ipynb](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment2/Assignment2.ipynb)

**README:** [Assignment 2 README](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment2/README.md)

### Assignment 3 — Feature Selection and Neural Networks

The third assignment focuses on **feature selection and neural network architectures**, covering both tabular and image classification tasks.

The implementation includes:

- Feature selection for Gaussian Naive Bayes classification.
- Variance-based feature filtering.
- Forward Selection and Backward Elimination.
- Evaluation using classification accuracy (CCR).
- Image classification using custom PyTorch neural networks.
- Linear and ReLU-based neural network architectures.
- An improved network using Batch Normalization, Dropout, and L2 regularization.
- Early Stopping for training control.
- Transfer learning with pretrained AlexNet and VGG16.
- Training and validation curve visualization and model comparison.

**Notebook:** [Assignment3.ipynb](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment3/Assignment3.ipynb)

**README:** [Assignment 3 README](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment3/README.md)

### Assignment 4 — Expectation-Maximization and Gaussian Mixture Models

The fourth assignment investigates **unsupervised learning and probabilistic clustering**, with an emphasis on Gaussian Mixture Models (GMMs) and the Expectation-Maximization (EM) algorithm.

The implementation includes:

- Comparing K-Means clustering with Gaussian Mixture Models.
- GMM parameter estimation using the Expectation-Maximization algorithm.
- Investigating sensitivity to initialization and local optima through repeated model fitting.
- Exploring singular covariance matrices and numerical instability.
- Comparing spherical, diagonal, tied, and full covariance structures.
- Generating handwritten digit samples using a Gaussian Mixture Model.
- Bayesian Gaussian Mixture Models and automatic component selection.
- Visualizing component weights and clustering results.

**Notebook:** [Assignment4.ipynb](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment4/Assignment4.ipynb)

**README:** [Assignment 4 README](https://github.com/MHAltafi/ML-coursework/blob/main/Assignment4/README.md)

## Repository Structure

```text
ML-coursework/
│
├── Assignment1/
│   ├── data/
│   ├── figs/
│   ├── Assignment1.ipynb
│   └── README.md
│
├── Assignment2/
│   ├── figs/
│   ├── Assignment2.ipynb
│   ├── normal.data.txt
│   └── README.md
│
├── Assignment3/
│   ├── .gitignore
│   ├── figs/
│   ├── Assignment3.ipynb
│   └── README.md
│
├── Assignment4/
│   ├── figs/
│   ├── Assignment4.ipynb
│   └── README.md
│
└── README.md
```

**Note:** The `data/` and `models/` directories in Assignment 3 are excluded from version control using `.gitignore`.

## Technologies

The coursework uses a range of Python tools and machine learning libraries, including:

- **Python**
- **Jupyter Notebook**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **SciPy**
- **Scikit-learn**
- **PyTorch**
- **torchvision**
- **torchmetrics**
- **torchsummary**
- **Pillow**
- **tqdm**

## Topics Covered

| Assignment | Area | Topics |
|---|---|---|
| Assignment 1 | Probabilistic Classification | Gaussian Discriminant Analysis, Shared and Class-Specific Covariance |
| Assignment 1 | Dimensionality Reduction | PCA, Decision Boundaries, Rejection Regions |
| Assignment 1 | Bayesian Classification | Gaussian Naive Bayes, Bag-of-Words, MLE, MAP, Dirichlet Smoothing |
| Assignment 2 | Density Estimation | Parzen Windows, Kernel Density Estimation, Bandwidth Selection |
| Assignment 2 | Statistical Inference | MLE, Asymptotic Covariance, Confidence Intervals, Hypothesis Testing |
| Assignment 2 | Nonlinear Regression | Gaussian Basis Functions, Ridge Regression, Bias-Variance Tradeoff |
| Assignment 3 | Feature Selection | Variance Thresholding, Forward Selection, Backward Elimination |
| Assignment 3 | Neural Networks | Linear Models, ReLU, Batch Normalization, Dropout, L2 Regularization |
| Assignment 3 | Deep Learning | PyTorch, Early Stopping, Transfer Learning, AlexNet, VGG16 |
| Assignment 4 | Unsupervised Learning | K-Means, Gaussian Mixture Models, Expectation-Maximization |
| Assignment 4 | Probabilistic Modeling | Covariance Structures, Singularities, Local Optima |
| Assignment 4 | Bayesian Learning | Bayesian Gaussian Mixture Models, Component Selection |
| Assignment 4 | Generative Models | Sampling, Synthetic Data Generation, Handwritten Digits |

---

*Coursework completed as part of the Machine Learning course at the University of Tehran.*