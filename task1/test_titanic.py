import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# for the test i need to take the data and manupilate it the same as my training data, meaning the male/female and Name nad Cabin and droping unwanted columns

gender_mapping = {'male': 1, 'female': 2}

# this indicates the survivor bvased on the Title. number 1: lowest survival/ 5:highest survival
survival_mapping = {
    'Mr': 1, 'Master': 3, 'Miss': 4, 'Mrs': 4, 'Dr': 2, 'Rev': 2, 'Col': 2, 'Major': 2, 'Mlle': 4, 'Ms': 4, 'Mme': 4, 'Lady': 5,      'Sir': 5, 'Countess': 5
}
# the deck which the Passenger Was in from the Cabin. number 1: lowest survival/ 5:highest survival
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

        features = torch.tensor(row[1:].values, dtype=torch.float32)
        target = torch.tensor(row.iloc[0], dtype=torch.float32).view(1)
        return features, target

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=8):
        super(NeuralNetwork, self).__init__()
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

device = 'cpu'


# read input file
datafile = "./Titanic-Dataset.xls"
init_data = pd.read_csv(datafile)

# clean it from unwanted data, and clean the Nan rows
filterd_dt = init_data.drop(columns=['PassengerId','Embarked', 'Ticket']).dropna()

# Use the Name and Cabin string as an integer representation
filterd_dt['survival_rate'] = init_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
filterd_dt['survival_rate'] = filterd_dt['survival_rate'].map(survival_mapping).fillna(1)
filterd_dt['Cabin'] = filterd_dt['Cabin'].str[0]
filterd_dt['Deck'] = filterd_dt['Cabin'].map(deck_values).fillna(1)

filterd_dt = filterd_dt.drop(columns=['Cabin', 'Name'])

filterd_dt['Sex'] = filterd_dt['Sex'].map(gender_mapping)

ds = TitanicDataSet(filterd_dt)

dl = DataLoader(ds, batch_size=len(ds), shuffle=False)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))


# this model accuracy ranges from 75%-85%
# some small test, some of it is from the titanic_survival dataset too
# featres are Pclass, Sex, Age, SibSp, Parch, fare, Survival_rate(takin from Name), Deck(takin from the Cabin first Char)


# as a test i Used the training set to get an output, its the same as getting the accuracy but it's to show that the modelk is saved and can be loaded later
model.eval()
with torch.no_grad():
    for b_f, b_t in dl:
        final_outputs = model(b_f)
    for i in range(len(final_outputs)):
            print(f"Output: {final_outputs[i].item():.0f}, Expected: {b_t[i].item():.0f}")