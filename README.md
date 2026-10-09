# AI & ML Internship — Task 5

## Decision Trees and Random Forests

### Overview

This project explores tree-based machine learning algorithms for classification using the Heart Disease dataset. It implements a Decision Tree Classifier and a Random Forest Classifier to study model performance, analyze overfitting, identify important features, and evaluate generalization using cross-validation.

### Objectives

- Train and visualize a Decision Tree Classifier.
- Analyze overfitting and control tree depth.
- Train a Random Forest Classifier and compare accuracy.
- Interpret feature importance scores.
- Evaluate model performance using cross-validation.

### Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Graphviz

### Dataset

The project uses `heart.csv`, which contains clinical attributes associated with heart disease.

- Original records: 1,025
- Records after duplicate removal: 302
- Input features: 13
- Target variable: `target`

Exact duplicate records are removed before training and evaluation. The target variable represents the presence or absence of heart disease according to the dataset labels.

### Implementation

**1. Decision Tree Classification**

Trained a Decision Tree Classifier and visualized its structure using Graphviz.

**2. Overfitting Analysis**

Compared training and testing accuracy at maximum depths of 1, 2, 3, 5, 10, and unlimited depth.

**3. Random Forest Classification**

Trained a Random Forest with 100 estimators and compared its testing accuracy with the Decision Tree.

**4. Feature Importance**

Calculated and plotted Random Forest feature importance scores to identify which features contributed most to the model's predictions.

**5. Cross-Validation**

Used five-fold cross-validation to evaluate the Random Forest across multiple data splits.

### Results

| Evaluation Metric                               | Accuracy |
| ----------------------------------------------- | -------: |
| Decision Tree — Training                        |  100.00% |
| Decision Tree — Testing                         |   80.33% |
| Random Forest — Testing                         |   75.41% |
| Random Forest — Average 5-Fold Cross-Validation |   82.78% |

The Decision Tree performed better than the Random Forest on the selected test split. Its perfect training accuracy and lower testing accuracy suggest possible overfitting.

### Feature Importance

The five highest-ranked features in the trained Random Forest were:

1. `cp`
2. `thalach`
3. `ca`
4. `oldpeak`
5. `thal`

Feature importance represents the model's relative use of each feature; it does not establish medical causation.

### Project Structure

```text
Task-05-Decision-Trees-Random-Forests/
├── dataset/
│   └── heart.csv
├── plots/
│   ├── decision_tree.dot
│   ├── decision_tree_graphviz.png
│   └── feature_importance.png
├── decision_tree_random_forest.py
└── README.md
```

### Installation and Execution

**Install the required Python libraries:**

```bash
pip install pandas matplotlib scikit-learn
```

Install Graphviz separately and ensure its executable path matches the path configured in the Python script.

**Run the project:**

```bash
python decision_tree_random_forest.py
```

### Key Learnings

- Understanding Decision Tree and Random Forest classifiers.
- Identifying possible overfitting by comparing training and testing accuracy.
- Visualizing decision trees using Graphviz.
- Interpreting feature importance scores.
- Evaluating models using five-fold cross-validation.

### Limitations

The results depend on the dataset, preprocessing, and data split. The dataset contains 302 unique records after duplicate removal, so further validation on independent data would be needed before drawing conclusions about real-world performance. This project is for educational purposes and is not intended for clinical diagnosis.

---

**Internship:** AI & ML Internship
**Task:** Task 5 — Decision Trees and Random Forests
