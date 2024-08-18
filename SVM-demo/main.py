import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# [shape, size, position]
# shape: 0 = nghỉ đen, 1 = nghỉ trắng

X = np.array([
    [0, 10, 1], [0, 12, 2], [0, 11, 1.5], [0, 10.5, 2.1], [0, 12.2, 1.8],
    [0, 11.1, 1.9], [0, 10.7, 1.7], [0, 12.3, 2.2], [0, 10.9, 2.1], [0, 11.5, 1.6],
    [1, 15, 1], [1, 16, 1.1], [1, 15.5, 1.2], [1, 15.2, 1.3], [1, 16.1, 1.4],
    [1, 15.8, 1.5], [1, 15.9, 1.6], [1, 16.2, 1.7], [1, 15.7, 1.8], [1, 16.5, 1.9]
])

y = np.array([0] * 10 + [1] * 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = svm.SVC(kernel='linear')  # kernel tuyến tính

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

new_samples = np.array([
    [0, 11, 2.0],
    [1, 15.8, 1.3]
])

predictions = model.predict(new_samples)
print(f"Predictions: {predictions}")
