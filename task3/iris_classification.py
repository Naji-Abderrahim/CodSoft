import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Categories
# ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
# features: petal_length, petal_width, sepal_length, sepal_width

# i can add Perimeter and Erea feature to help with the classification
map_c = {
    'Iris-setosa': 'red',
    'Iris-versicolor': 'blue',
    'Iris-virginica': 'green'
    }

df = pd.read_csv('./IRIS.xls')

Y = df['species']
X = df.drop(columns=['species'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

classifer = LogisticRegression(random_state=42, max_iter=500)

model = classifer.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(model.score(x_test, y_test))
