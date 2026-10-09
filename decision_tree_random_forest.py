import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/heart.csv")

print("Original dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)


print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
df.info()
X = df.drop(columns=["target"])
y = df["target"]

print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nDecision Tree trained successfully!")
from sklearn.metrics import accuracy_score

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\nTraining accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)

from sklearn.tree import export_graphviz
import subprocess

export_graphviz(
    model,
    out_file="plots/decision_tree.dot",
    feature_names=X.columns,
    class_names=["No Heart Disease", "Heart Disease"],
    filled=True,
    rounded=True,
    max_depth=3
)



subprocess.run(
    [
        r"C:\Program Files\Graphviz\bin\dot.exe",
        
        "-Tpng",
        "plots/decision_tree.dot",
        "-o",
        "plots/decision_tree_graphviz.png"
    ],
    check=True
)
print("\nGraphviz decision tree saved successfully!")


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\n--- Decision Tree Depth Comparison ---")

for depth in [1, 2, 3, 5, 10, None]:

    depth_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    depth_model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train, depth_model.predict(X_train)
    )
    test_acc = accuracy_score(
        y_test, depth_model.predict(X_test)
    )

    print(
        f"Depth: {str(depth):4} | "
        f"Training Accuracy: {train_acc:.4f} | "
        f"Testing Accuracy: {test_acc:.4f}"
    )
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n--- Random Forest Results ---")
print("Random Forest Testing Accuracy:", rf_accuracy)

dt_predictions = model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\n--- Model Accuracy Comparison ---")
print("Decision Tree Testing Accuracy:", dt_accuracy)
print("Random Forest Testing Accuracy:", rf_accuracy)




importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Feature Importances ---")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Random Forest Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "plots/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nFeature importance plot saved successfully!")

from sklearn.model_selection import cross_val_score

print("\n--- Cross-Validation Results ---")

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Scores for each fold:", cv_scores)
print("Average Cross-Validation Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())
print("Duplicate rows:", df.duplicated().sum())