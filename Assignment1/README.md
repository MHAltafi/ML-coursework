# Machine Learning – Assignment 1

This assignment implements and evaluates probabilistic classification methods using Python.

## Assignment Overview

The assignment covers three classification problems:

- **Question 6:** Gaussian Discriminant Analysis (GDA)
- **Question 7:** Gaussian Naive Bayes
- **Question 8:** Bag of Words Naive Bayes

## Requirements

The project uses Python and the following libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

Install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

## Question 6 – Gaussian Discriminant Analysis

The Iris dataset is used, with PCA reducing its four features to two dimensions.

The experiments include:
- Gaussian parameter estimation with shared and class-specific covariance matrices.
- Decision boundaries and the effect of class priors.
- Three-class classification.
- Training with 80% of the dataset.
- Classification with a rejection option.

**Decision boundaries with different class priors:**

![GDA Decision Boundaries](figs/gda_decision_boundaries.png)

**Three-class classification regions:**

![GDA Three Classes](figs/gda_three_classes.png)

**Classification with rejection for different rejection costs:**

![GDA Rejection Regions](figs/gda_rejection_regions.png)

## Question 7 – Gaussian Naive Bayes

A Gaussian Naive Bayes classifier is implemented from scratch using the provided dataset.

The implementation includes:
- Class prior estimation.
- Gaussian likelihood estimation.
- Classification using log-probabilities.
- Evaluation using accuracy, precision, and recall.

**Test accuracy: 83.5%**

**Confusion matrix:**

![Gaussian Naive Bayes Confusion Matrix](figs/gaussian_nb_confusion_matrix.png)

## Question 8 – Bag of Words Naive Bayes

A binary text classifier is developed to distinguish between Baseball and Hockey documents using Bag of Words representations.

The experiments compare:
- Maximum Likelihood Estimation (MLE).
- Maximum A Posteriori (MAP) with Dirichlet smoothing.
- Different smoothing parameters (β).
- Different training set sizes.

**Experimental results:**

| Method | Accuracy |
|---|---:|
| MLE | 50% |
| Best MAP | 87% |
| Best training-size experiment | 88% |

**Effect of smoothing parameter:**

![Beta Experiment](figs/naive_bayes_beta.png)

**Effect of training set size:**

![Training Size Experiment](figs/naive_bayes_training_size.png)

## Repository Structure

```text
Assignment1/
├── README.md
├── Assignment1.ipynb
├── figs/
│   ├── gda_decision_boundaries.png
│   ├── gda_three_classes.png
│   ├── gda_rejection_regions.png
│   ├── gaussian_nb_confusion_matrix.png
│   ├── naive_bayes_beta.png
│   └── naive_bayes_training_size.png
└── data/
    ├── guassianBayes.mat
    ├── baseball_train_set.csv
    ├── baseball_test_set.csv
    ├── hockey_train_set.csv
    ├── hockey_test_set.csv
    └── vocabulary.txt
```

## How to Run

1. Clone or download the repository.
2. Install the required dependencies.
3. Place all datasets inside the `data/` directory.
4. Open `Assignment1.ipynb` using Jupyter Notebook or JupyterLab.
5. Run all cells sequentially to reproduce the experiments and generate the figures.

```bash
pip install numpy pandas matplotlib scikit-learn scipy
jupyter notebook Assignment1.ipynb
```

## Conclusion

This assignment explores Gaussian Discriminant Analysis, Gaussian Naive Bayes, and Bag of Words classification. The experiments demonstrate the effects of class priors, rejection costs, probability smoothing, and training set size on classification performance.