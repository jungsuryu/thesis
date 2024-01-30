import os
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action = 'ignore')

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------------------------------------------------------

## 데이터 불러오기
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
movie_cols = ['item_id', 'title', 'genre']
user_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']

ratings = pd.read_csv('/home/ryu/thesis/data/ml-1m/ratings.dat', sep='::', names=ratings_cols, engine='python', encoding = "ISO-8859-1")
movies = pd.read_csv('/home/ryu/thesis/data/ml-1m/movies.dat', sep='::', names=movie_cols, engine='python', encoding = "ISO-8859-1")
users = pd.read_csv('/home/ryu/thesis/data/ml-1m/users.dat', sep='::', names=user_cols, engine='python', encoding = "ISO-8859-1")

## 평점 없는 영화 제거
movies_in_rating = ratings['item_id'].unique()
movies = movies[movies['item_id'].isin(movies_in_rating)]

## 장르 정리
genres_df = movies['genre'].str.get_dummies(sep='|')
movies = pd.concat([movies, genres_df], axis=1)
# movies = movies.drop(['genre'], axis=1)

## 필요한 정보만 추출
users = users[['user_id', 'age', 'sex', 'occupation']]


# 데이터 합병
data = pd.merge(ratings, movies, how='inner', on='item_id')
data = pd.merge(data, users, how='inner', on='user_id')
data['age'] /= 50

x = data.copy()
y = data['user_id']
ratings_train, ratings_test = train_test_split(x, test_size=0.25, stratify=y, random_state=8)

user_to_index = {user: idx for idx, user in enumerate(data['user_id'].unique())}
item_to_index = {item: idx for idx, item in enumerate(data['item_id'].unique())}
occ_to_index = {occ: idx for idx, occ in enumerate(data['occupation'].unique())}
sex_to_index=  {gender: idx for idx, gender in enumerate(data['sex'].unique())}

ratings_train['user_id_conv'] = ratings_train.user_id.map(user_to_index)
ratings_train['item_id_conv'] = ratings_train.item_id.map(item_to_index)
ratings_train['occ_id_conv'] = ratings_train.occupation.map(occ_to_index)
ratings_train['sex_id_conv'] = ratings_train.sex.map(sex_to_index)

ratings_test['user_id_conv'] = ratings_test.user_id.map(user_to_index)
ratings_test['item_id_conv'] = ratings_test.item_id.map(item_to_index)
ratings_test['occ_id_conv'] = ratings_test.occupation.map(occ_to_index)
ratings_test['sex_id_conv'] = ratings_test.sex.map(sex_to_index)

# Dataset 생성
class CustomDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings, occupation, gender, genre, age):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)
        self.occupation = torch.tensor(occupation, dtype=torch.long)
        self.gender = torch.tensor(gender, dtype=torch.long)
        self.genre = torch.tensor(genre, dtype=torch.long)
        self.age = torch.tensor(age, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx], self.occupation[idx], self.gender[idx], self.genre[idx], self.age[idx]
    

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_occ, num_layers, embedding_size, num_factors, dropout=None):
        super().__init__()

        self.dropout = dropout

        # FM part
        self.user_embedding_fm = nn.Embedding(num_users, num_factors)
        self.item_embedding_fm = nn.Embedding(num_items, num_factors)
        self.occ_embedding_fm = nn.Embedding(num_occ, num_factors)
        # self.gender_embedding_fm = nn.Embedding(num_gen, num_factors)

        # self.w_0 = nn.Parameter(torch.zeros(1))  # global bias
        self.w = nn.Parameter(torch.Tensor(num_factors * 3 + 18 + 1 + 1))  # 특성별 가중치
        self.v = nn.Parameter(torch.Tensor(num_factors * 3 + 18 + 1 + 1, embedding_size))  # 잠재 요인 가중치

        # MLP part
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size * (2 ** (num_layers - 1)))
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size * (2 ** (num_layers - 1)))
        self.occ_embedding_mlp = nn.Embedding(num_occ, embedding_size * (2 ** (num_layers - 1)))
        # self.gender_embedding_mlp = nn.Embedding(num_gen, embedding_size * (2 ** (num_layers - 1)))

        mlp_input_size = embedding_size * (2 ** (num_layers - 1)) * 3 + 20

        layers = []
        for i in range(num_layers):
            output_size = embedding_size * (2 ** (num_layers - i - 1))
            if dropout:
                layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Linear(mlp_input_size, output_size))
            layers.append(nn.ReLU())
            mlp_input_size = output_size
        
        self.mlp_layers = nn.Sequential(*layers)

        # Final prediction layer
        # final_layer_input_size = embedding_size + output_size + text_emb_size
        self.final_layer = nn.Linear(222, 1)

        self._init_weight_()

    def _init_weight_(self):
        # Initialize weights here
        nn.init.normal_(self.user_embedding_fm.weight, std=0.01)
        nn.init.normal_(self.item_embedding_fm.weight, std=0.01)
        nn.init.normal_(self.occ_embedding_fm.weight, std=0.01)
        # nn.init.normal_(self.gender_embedding_fm.weight, std=0.01)

        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.occ_embedding_mlp.weight, std=0.01)
        # nn.init.normal_(self.gender_embedding_mlp.weight, std=0.01)
        

        nn.init.normal_(self.w, std=0.01)
        nn.init.normal_(self.v, std=0.01)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(self.final_layer.weight)
        
        nn.init.kaiming_normal_(self.final_layer.weight)
    
    def forward(self, user_input, item_input, occ_input, gender_input, genre_input, age_input):
        # FM part
        user_emb_mf = self.user_embedding_fm(user_input)
        item_emb_mf = self.item_embedding_fm(item_input)
        occ_emb_mf = self.occ_embedding_fm(occ_input)

        #print(user_emb_mf.shape, item_emb_mf.shape, occ_emb_mf.shape, genre_input.shape, gender_input.shape, age_input.shape)
        gender_input = gender_input.unsqueeze(1)
        age_input = age_input.unsqueeze(1)
        x = torch.cat([user_emb_mf, item_emb_mf, occ_emb_mf, gender_input, genre_input, age_input],dim=1)
        # print(x.shape)

        # 1차 상호작용: 각 사용자와 아이템의 가중치를 곱하고, 그 결과를 모두 더함
        linear_terms = torch.sum(x * self.w, dim=1)

        # print(linear_terms.shape)

        # 2차 상호작용
        interactions = 0.5 * torch.sum(
            torch.pow(torch.matmul(x, self.v), 2) - torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2)), dim=1)


        # 예측값 계산
        # predict = self.w_0 + linear_terms + interactions
        linear_terms = linear_terms.unsqueeze(-1)
        interactions = interactions.unsqueeze(-1)
        predict = torch.cat([linear_terms, interactions], dim=1)

        # pred2 = nn.Linear(2, 220)(predict)

        # predict = predict.unsqueeze(-1)
        # print(predict.shape)


        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_input)
        item_emb_mlp = self.item_embedding_mlp(item_input)
        occ_emb_mlp = self.occ_embedding_mlp(occ_input)
        mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp, occ_emb_mlp, gender_input, genre_input, age_input], dim=1)
        mlp_vector = self.mlp_layers(mlp_vector)

        # print(mlp_vector.shape)

        vector = torch.cat([predict, mlp_vector],dim=1)
        # vector = torch.cat([pred2, mlp_vector], dim=1)

        # print(vector.shape)
    

        rating = self.final_layer(vector)
        return rating.squeeze()



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0005):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler=None, patience=5):
    early_stopping = EarlyStopping(patience=patience)
    train_rmse_hist = []
    val_rmse_hist = []

    best_val_rmse = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, item, rating, occ, gender, genre, age in train_loader:
            user, item, rating, occ, gender, genre, age = user.to(device), item.to(device), rating.float().to(device), occ.to(device), gender.to(device), genre.to(device), age.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, item, occ, gender, genre, age)
            loss = criterion(prediction.view(-1), rating.view(-1))
            # loss = criterion(prediction, rating)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        train_rmse = torch.sqrt(torch.tensor(avg_loss))
        train_rmse_hist.append(train_rmse.item())

        val_rmse = evaluate(model, criterion, val_loader)
        val_rmse_hist.append(val_rmse.item())

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}, Validation RMSE: {val_rmse:.6f}")
        
        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    print(f"Best Validation RMSE: {best_val_rmse:.8f} at Epoch {best_epoch}")
    return train_rmse_hist, val_rmse_hist, best_val_rmse


def evaluate(model, criterion, val_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for user, item, rating, occ, gender, genre, age in val_loader:
            user, item, rating, occ, gender, genre, age = user.to(device), item.to(device), rating.float().to(device), occ.to(device), gender.to(device), genre.to(device), age.to(device)
            prediction = model(user, item, occ, gender, genre, age)
            loss = criterion(prediction.view(-1), rating.view(-1))
            # loss = criterion(prediction, rating)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return torch.sqrt(torch.tensor(avg_loss))

# 하이퍼파라미터 설정
num_users = len(set(data.user_id)) + 1
num_items = len(set(data.item_id)) + 1
num_layers= 4
embedding_size = 220
num_factors = 220
num_occ = len(set(data.occupation)) + 1
epochs = 100
batch_size = 1024
patience = 3


# 데이터 로더 정의 
train_dataset = CustomDataset(
    user_ids = ratings_train['user_id_conv'].values,
    item_ids = ratings_train['item_id_conv'].values,
    ratings = ratings_train['rating'].values.astype(np.float32),
    occupation = ratings_train['occ_id_conv'].values,
    gender = ratings_train['sex_id_conv'].values,
    genre = ratings_train.iloc[:, 6:24].values,
    age = ratings_train['age'].values
)

test_dataset = CustomDataset(
    user_ids = ratings_test['user_id_conv'].values,
    item_ids = ratings_test['item_id_conv'].values,
    ratings = ratings_test['rating'].values.astype(np.float32),
    occupation = ratings_test['occ_id_conv'].values,
    gender = ratings_test['sex_id_conv'].values,
    genre = ratings_test.iloc[:, 6:24].values,
    age = ratings_test['age'].values
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=4) # 남바 오브 와카ㅡ스, 근데 배치를 키우셔야할거같은데여
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


# 모델 인스턴스 생성
model = NeuMF(num_users, num_items, num_occ, num_layers, embedding_size, num_factors)
model = nn.DataParallel(model).cuda()

# 손실 함수와 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0014, weight_decay=0.015)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model.to(device)
criterion.to(device)



train_rmse, val_rmse, best_RMSE = train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs, scheduler, patience)
print(f'Result: {best_RMSE.item()}')