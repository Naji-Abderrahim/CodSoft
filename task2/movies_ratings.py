from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from dataset import Data

# form this problem i did try linear-regression method Y = aX + but didn't get any good results
# So i opt into a tree-like regressor model and with this get good results

df = Data('./movies_dataset.csv').initiate
Y = df['Rating']
X = df.drop(columns=['Rating'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(f"X_tr {len(x_train)} X_test {len(x_test)}")

tree = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
)

tree.fit(x_train, y_train)

score = tree.score(x_test, y_test)
print(f'Model\'s Score {score}')
