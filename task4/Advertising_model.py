import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./advertising.xls")
df['Sum'] = df['TV'] + df['Radio'] + df['Newspaper']
Y = df['Sales']
X = df.drop(columns=['Sales'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
linearRegression = LinearRegression()
model = linearRegression.fit(x_train, y_train)
y_pred = model.predict(x_test)


print(model.score(x_test, y_test))

f, axes = plt.subplots(2, 2, figsize=(10, 6))
axes[0][0].scatter(x_test['TV'], y_test)
axes[0][0].set_title('Act')
axes[0][1].scatter(x_test['TV'], y_pred)
axes[0][1].set_title('pred')
plt.show()

