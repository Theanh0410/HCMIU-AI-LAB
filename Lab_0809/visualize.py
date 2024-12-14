import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import MultiClassPerceptron

# Load the data
filepath = 'iris/iris.data'
df = pd.read_csv(filepath, sep=',', header=None, names=['sepal_len', 'sepal_wid', 'pedal_len', 'pedal_wid', 'class'])

print(df.head())

# Map class labels and extract features
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
class_mapping_2classes = {'Iris-setosa': 0, 'Iris-versicolor': 1}
df['class'] = df['class'].map(class_mapping)

X = df[['sepal_len', 'sepal_wid']].values
y = df['class'].values

# Randomize the dataset
data_randomized = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(data_randomized.head())

training_test_index = round(len(data_randomized) * 0.8)

# Training/Test split
training_set = data_randomized[:training_test_index]
test_set = data_randomized[training_test_index:]

# Extract features and labels
X_train = training_set[['sepal_len', 'sepal_wid']].values
y_train = training_set['class'].values
X_test = test_set[['sepal_len', 'sepal_wid']].values
y_test = test_set['class'].values

# Train the custom multi-class Perceptron
n_classes = len(np.unique(y))
multi_class_perceptron = MultiClassPerceptron(n_classes=n_classes)
multi_class_perceptron.fit(X_train, y_train)

# Evaluate the model
y_train_pred = multi_class_perceptron.predict(X_train)
y_test_pred = multi_class_perceptron.predict(X_test)

train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")    
    
color_map = {
    0: 'blue',      # Iris-setosa
    1: 'green',     # Iris-versicolor
    2: 'yellow'     # Iris-virginica
}
    
def visualize_sepal_length_width(X, y, class_mapping, color_map):
    plt.figure(figsize=(8, 6))

    for class_name, class_value in class_mapping.items():
        plt.scatter(
            X[y == class_value, 0],  # Sepal length
            X[y == class_value, 1],  # Sepal width
            label=class_name,
            color=color_map[class_value],
            alpha=0.8
        )

    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Visualization of Sepal Length and Sepal Width")
    plt.legend(title="Classes")
    plt.show()

visualize_sepal_length_width(X, y, class_mapping, color_map)
visualize_sepal_length_width(X, y, class_mapping_2classes, color_map)
