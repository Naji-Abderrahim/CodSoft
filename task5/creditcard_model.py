import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("./creditcard.csv")

Y = df['Class']
X = df.drop(columns=['Class'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

lr = LogisticRegression(max_iter=1000)

model = lr.fit(x_train, y_train)

y_pred = model.predict(x_test)

f, axes = plt.subplots(2, 2, figsize=(10, 6))
axes[0][0].scatter(x_test['V1'], y_test)
axes[0][0].set_title('Act')
axes[0][1].scatter(x_test['V1'], y_pred)
axes[0][1].set_title('pred')
plt.show()

print(model.score(x_test, y_test))
