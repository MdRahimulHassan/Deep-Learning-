#Dataset link: https://www.kaggle.com/code/sandragracenelson/lung-cancer-prediction

import math as m
import pandas as pd


def sigmoid(z):
    return 1 / (1 + m.exp(-z))

df = pd.read_csv("/content/survey lung cancer.csv")

df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})

df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})

X = df.drop("LUNG_CANCER", axis=1).values
y = df["LUNG_CANCER"].values

n_features = X.shape[1]

weights = [0.1] * n_features
b = 0.0

lr = 0.001

for epoch in range(2001):
    total_loss = 0

    for i in range(len(X)):
        x = X[i]
        target = y[i]

        z = sum(x[j] * weights[j] for j in range(n_features)) + b
        prediction = sigmoid(z)

        error = prediction - target
        loss = error ** 2
        total_loss += loss

        d_pred = prediction * (1 - prediction)

        for j in range(n_features):
            dw = error * d_pred * x[j]
            weights[j] -= lr * dw

        db = error * d_pred
        b -= lr * db

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d} loss {total_loss:.6f}")

print("\nFinal weights:")
for i, w in enumerate(weights):
    print(f"w{i+1} = {w:.4f}")
print(f"b = {b:.4f}")

print("\nPredictions:")
for i in range(len(X)):
    x = X[i]
    target = y[i]

    z = sum(x[j] * weights[j] for j in range(n_features)) + b
    prediction = sigmoid(z)
    loss = (prediction - target) ** 2

    print(f"x={x[:5]}... target={target} pred={prediction:.4f} loss={loss:.6f}")