import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd

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


def train(dataloader, model, loss_fn, optimizer, epochs=1500):
    for ep in range(epochs):
        for x, y in dataloader:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if ep % 100 == 0:
            print(f"epoch {ep} loss = {loss}")


# read input file
datafile = "./Titanic-Dataset.xls"
init_data = pd.read_csv(datafile)

# clean it from unwanted data, and clean the Nan rows
filterd_dt = init_data.drop(columns=['PassengerId', 'Embarked', 'Ticket']).dropna()

# Use the Name and Cabin string as an integer representation
filterd_dt['survival_rate'] = init_data['Name'].str.extract(' ([A-Za-z]+)', expand=False)
filterd_dt['survival_rate'] = filterd_dt['survival_rate'].map(survival_mapping).fillna(1)
filterd_dt['Cabin'] = filterd_dt['Cabin'].str[0]
filterd_dt['Deck'] = filterd_dt['Cabin'].map(deck_values).fillna(1)

filterd_dt = filterd_dt.drop(columns=['Cabin', 'Name'])

filterd_dt['Sex'] = filterd_dt['Sex'].map(gender_mapping)

Y = filterd_dt['Survived']
X = filterd_dt.drop(columns=['Survived'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

train_df = pd.concat([x_train, y_train], axis=1)
test_df = pd.concat([x_test, y_test], axis=1)

train_ds = TitanicDataSet(train_df)
test_ds = TitanicDataSet(test_df)

train_dl = DataLoader(train_ds, batch_size=20, shuffle=False)
train_dl_acc = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

model = TitanicNN()

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

print("Model is training...")
train(train_dl, model, loss_fn, optimizer)
print("Done!")

torch.save(model.state_dict(), "./model.pth")
print("Model Saved to model.pth")

model.eval()
with torch.no_grad():
    for b_f, b_t in train_dl_acc:
        final_outputs = model(b_f)
    predictions = (final_outputs > 0.5).float()
    correct = (predictions == b_t).float().sum()
    accuracy = correct / b_t.numel()
    print(f"Accuracy at train set: {accuracy.item() * 100:.2f}%")
    for b_f, b_t in test_dl:
        final_outputs = model(b_f)
    predictions = (final_outputs > 0.5).float()
    correct = (predictions == b_t).float().sum()
    accuracy = correct / b_t.numel()
    print(f"Accuracy at test set: {accuracy.item() * 100:.2f}%")
