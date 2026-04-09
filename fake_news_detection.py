import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Step 1: Load datasets
true_df = pd.read_csv("data/true.csv")
false_df = pd.read_csv("data/fake.csv")

# Step 2: Add labels (1 = Real, 0 = Fake)
true_df["label"] = 1
false_df["label"] = 0

# Step 3: Combine both datasets
data = pd.concat([true_df, false_df], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle rows

print("Dataset shape:", data.shape)
print(data.head())

# Step 4: Handle missing values
data = data.fillna('')

# Step 5: Combine title and text (if both exist)
if "title" in data.columns and "text" in data.columns:
    data["content"] = data["title"] + " " + data["text"]
else:
    # if only text column exists
    data["content"] = data["text"]

# Step 6: Split input and output
X = data["content"]
y = data["label"]

# Step 7: Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Step 8: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 9: Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Step 10: Evaluate the model
y_pred = model.predict(X_test_vectorized)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 11: Visualization - Confusion Matrix (Figure 3.1)
cm = confusion_matrix(y_test, y_pred)

# Create folder for saving figures if not already there
os.makedirs("figures", exist_ok=True)

# Plot the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title("Figure 3.1: Confusion Matrix - Fake News Detection", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Save figure as PNG inside a folder
save_path = os.path.join("figures", "Figure3_1_ConfusionMatrix.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Confusion Matrix image saved successfully at: {os.path.abspath(save_path)}")

# Step 12: Calculate training accuracy
train_pred = model.predict(X_train_vectorized)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

# Step 13: Plot accuracy comparison
plt.figure(figsize=(6,4))
plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_acc*100, test_acc*100])
plt.title('Training vs Testing Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, v in enumerate([train_acc*100, test_acc*100]):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)
plt.savefig('figure_3_2_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
