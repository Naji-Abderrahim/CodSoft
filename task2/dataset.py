import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


# data analysing:
# Name: not usfel (in the regession case)
# Year: as i observe the data that i have the olfer the movie is the higher the ratings are
# Duration: the less the Movie's Duration is the more rating the movie gets
# Genre: abit complicated but spliting the Genre and plotting each Genre/Rating i have the resault that i can use to recreate a intger Genre representation
# Director/Actors: for this version i don't see any use case for these ones so i will drop them
# data scaling
# Scale all the Value so all value will have the same impact on the model using 'log(n)' to Scale Dowmn Duration and Year and Votes
# Genre is fixed, each type has its own column with an integer indicating the presence of thisgenre
# Create the DataSet Class


class Data:
    def __init__(self, dataPath):
        self.df = pd.read_csv(dataPath, encoding='latin1').drop(
            columns=['Name', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])
        imputer = SimpleImputer(strategy='median')  # or 'median'
        self.df['Rating'] = imputer.fit_transform(self.df['Rating'].values.reshape(-1, 1))
        # self.df = self.df[self.df['Rating'].notna()]

    def prepare(self):
        self.df = self.df[self.df['Votes'] != '$5.16M']

        self.df['Year'] = self.df['Year'].dropna().str.replace(
            '(', '', regex=False).str.replace(')', '', regex=False).astype(float)
        self.df['Duration'] = self.df['Duration'].dropna(
        ).str.replace(' min', '', regex=True).astype(float)
        self.df['Votes'] = self.df['Votes'].dropna().str.replace(
            ',', '', regex=False).astype(float)

        y_med = self.df['Year'].dropna().median()
        d_med = self.df['Duration'].dropna().median()
        v_med = self.df['Votes'].dropna().median()

        self.df['Year'] = self.df['Year'].fillna(y_med)
        self.df['Duration'] = self.df['Duration'].fillna(d_med)
        self.df['Votes'] = self.df['Votes'].fillna(v_med)

        self.df['Year'] = self.df['Year'].apply(lambda x: np.log(x))
        self.df['Duration'] = self.df['Duration'].apply(lambda x: np.log(x))
        self.df['Votes'] = self.df['Votes'].apply(lambda x: np.log(x))
        self.df['Genre'] = self.df['Genre'].fillna('')

    def handleGenre(self):
        uniqueGenre = list(set(self.df['Genre'].str.split(
            ',').explode().str.strip().astype(str)))
        uniqueGenre.remove('')
        for g in uniqueGenre:
            self.df[g] = self.df['Genre'].apply(lambda x: 1 if x in g else 0)
        self.df = self.df.drop(columns=['Genre'])

    @property
    def initiate(self):
        # return self.df
        self.prepare()
        self.handleGenre()
        return self.df


class MovieDatset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        t = row['Rating']
        f = row.drop('Rating')
        features = torch.tensor(f.values, dtype=torch.float32)
        target = torch.tensor(t, dtype=torch.float32).view(1)
        return features, target


# Data Tests
# df = pd.read_csv("movies_dataset.csv", encoding='latin1')

# df = df.dropna()

# # df['Duration'] = df['Duration'].str.replace(' min', '', regex=False).astype(float)
# df['Year'] = df['Year'].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(int)

# # data Scaling Z-score algorithmn applyed to Year, Duration, Votes
# d_mean = df['Year'].mean()
# d_std = df['Year'].std()
# y1 = df['Year'].apply(lambda x: (x - d_mean) / d_std)
# df['Votes'] = df['Votes'].str.replace(',', '', regex=False).astype(int)
# # print(set(df['Votes']))
# # d_mean = df['Votes'].astype(float).mean()
# # d_std = df['Votes'].astype(float).std()
# # df['new_Votes'] = df['Votes'].apply(lambda x: (x - d_mean) / d_std)
# x = df['Rating']
# # y1 = df['Year']
# y2 = df['Year'].apply(lambda x: np.log(x))
# y3 = df['Votes'].apply(lambda x: np.log(x))

# # Genre spliting an binary indicating each type
# gnr = df['Genre'].str.split(',')
# # status = gnr.explode().str.strip()
# # unique_gnr = list(set(status))
# # for g in unique_gnr:
# #     df[g] = df['Genre'].apply(lambda x: int (g in x))
# # df = df.drop(columns=['Genre'])
# uniqueGenre = list(set(df['Genre'].str.split(',').explode().str.strip()))
# for g in uniqueGenre:
#     df[g] = df['Genre'].apply(lambda x: int (g in x))
# df = df.drop(columns=['Genre'])
# print(df)

# # genre = gnr.explode().str.strip()
# # n_df = pd.DataFrame({
# #     'Genre': genre,
# #     'Rating': df.loc[genre.index, 'Rating']
# # })


# # x = n_df['Rating'].sort_values(ascending=False)
# # y = n_df['Genre'].sort_values(ascending=True)
# # plt.scatter(x, y)
# # x = df['Rating']
# # y1 = df["Duration"].sort_values(ascending=False)
# # y3 = df["new_Duration"].sort_values(ascending=False)
# # y2 = df['Year'].sort_values(ascending=False)
# # y4 = df['new_Year'].sort_values(ascending=False)
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
# axes[0].scatter(x, y1, color='blue', label='Year')
# axes[1].scatter(x, y2, color='red', label='Year')
# # plt.scatter(x, y4, color='red', label='new_Year')
# # plt.legend()
# axes[0].grid(True)
# axes[1].grid(True)
# plt.show()
