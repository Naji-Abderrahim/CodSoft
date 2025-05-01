import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

gender_mapping = {'male': 1, 'female': 2}
survival_mapping = {
    'Mr': 1,
    'Master': 3,
    'Miss': 4,
    'Mrs': 4,
    'Dr': 2,
    'Rev': 2,
    'Col': 2,
    'Major': 2,
    'Mlle': 4,
    'Ms': 4,
    'Mme': 4,
    'Lady': 5,
    'Sir': 5,
    'Countess': 5
}
deck_values = {
    'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0
}


class TitanicDataSet(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        t = row['Survived']
        f = row.drop('Survived')

        features = torch.tensor(f.values, dtype=torch.float32, requires_grad=True)
        target = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(1)
        return features, target


class TitanicNN(nn.Module):
    def __init__(self, input_size=8):
        super(TitanicNN, self).__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.bn1(torch.relu(self.l1(x)))
        x = self.dropout(x)
        x = self.bn2(F.leaky_relu(self.l2(x), 0.01))
        x = self.dropout(x)
        x = torch.sigmoid(self.l3(x))
        return x


datafile = "./Titanic-Dataset.xls"
init_data = pd.read_csv(datafile)

filterd_dt = init_data.drop(columns=['PassengerId', 'Embarked', 'Ticket']).dropna()

filterd_dt['survival_rate'] = init_data['Name'].str.extract(' ([A-Za-z]+)', expand=False)
filterd_dt['survival_rate'] = filterd_dt['survival_rate'].map(survival_mapping).fillna(1)
filterd_dt['Cabin'] = filterd_dt['Cabin'].str[0]
filterd_dt['Deck'] = filterd_dt['Cabin'].map(deck_values).fillna(1)
filterd_dt['Sex'] = filterd_dt['Sex'].map(gender_mapping)

filterd_dt = filterd_dt.drop(columns=['Cabin', 'Name'])

Y = filterd_dt['Survived']
X = filterd_dt.drop(columns=['Survived'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

train_df = pd.concat([x_train, y_train], axis=1)
test_df = pd.concat([x_test, y_test], axis=1)

train_ds = TitanicDataSet(train_df)
test_ds = TitanicDataSet(test_df)

train_dl = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

model = TitanicNN()
print("Loading saved model...")
model.load_state_dict(torch.load("model.pth", weights_only=True))
x_axis = x_test['Age']

model.eval()
with torch.no_grad():
    for ftrs, trgts in test_dl:
        y_pred = model(ftrs)
        y_pred = (y_pred > 0.5).float()
    fig, axis = plt.subplots(1, 2, figsize=(15, 8))
    axis[0].scatter(x_axis, trgts)
    axis[0].set_title("Actual Data")
    axis[0].set_xlabel("Age")
    axis[0].set_ylabel("Survived")
    axis[1].scatter(x_axis, y_pred)
    axis[1].set_title("predected results")
    axis[1].set_xlabel("Age")
    axis[1].set_ylabel("Survived")
    plt.show()
