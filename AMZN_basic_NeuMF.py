# 필요한 모듈 불러오기
import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action = 'ignore')

import pickle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset 생성
'''
데이터셋 정의 역할. 데이터에 대한 정보 포함하고 개별 데이터에 대한 접근 방법 제공.
하지만 직접 데이터를 로드하진 않음. 대신 DataLoader에 전달하여 실제 데이터 로딩과 미니배치 생성, 셔플링, 멀티스레드 로딩 등의 기능 수행
'''
class CustomDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]
    

# Define the NeuMF model with weight initialization
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors, num_layers, dropout=None):
        super().__init__()
        
        self.dropout = dropout

        # GMF part
        self.user_gmf_embedding = nn.Embedding(num_users, num_factors)
        self.item_gmf_embedding = nn.Embedding(num_items, num_factors)
        
        # MLP part
        self.user_mlp_embedding = nn.Embedding(num_users, num_factors * (2 ** (num_layers - 1)))
        self.item_mlp_embedding = nn.Embedding(num_items, num_factors * (2 ** (num_layers - 1)))

        layers = []
        for i in range(num_layers):
            input_size = num_factors * (2 ** (num_layers-i))
            if dropout:
                layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Linear(input_size, input_size//2))
            layers.append(nn.ReLU())
        
        self.mlp_layers = nn.Sequential(*layers)

        # Final prediction layer
        self.final_layer = nn.Linear(num_factors*2, 1)
        
        self._init_weight_()
        
    def _init_weight_(self):
        # Initialize weights here
        nn.init.normal_(self.user_gmf_embedding.weight, std=0.01)
        nn.init.normal_(self.item_gmf_embedding.weight, std=0.01)
        nn.init.normal_(self.user_mlp_embedding.weight, std=0.01)
        nn.init.normal_(self.item_mlp_embedding.weight, std=0.01)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight) # He initialization
                
        nn.init.kaiming_normal_(self.final_layer.weight)
        
    def forward(self, user_indices, item_indices):
        # GMF part
        user_gmf_embedding = self.user_gmf_embedding(user_indices)
        item_gmf_embedding = self.item_gmf_embedding(item_indices)
        gmf_vector = torch.mul(user_gmf_embedding, item_gmf_embedding)
        
        # MLP part
        user_mlp_embedding = self.user_mlp_embedding(user_indices)
        item_mlp_embedding = self.item_mlp_embedding(item_indices)
        mlp_vector = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        # Combine both parts
        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        
        # Final prediction
        rating = self.final_layer(vector)
        return rating.squeeze()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
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
        for user, item, rating in train_loader:
            user, item, rating = user.to(device), item.to(device), rating.float().to(device)
            
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction.view(-1), rating.view(-1))
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
        for user, item, rating in val_loader:
            user, item, rating = user.to(device), item.to(device), rating.float().to(device)
            prediction = model(user, item)
            loss = criterion(prediction.view(-1), rating.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return torch.sqrt(torch.tensor(avg_loss))




# 데이터 불러오기
ratings = pd.read_csv('/home/ryu/thesis/data/amazon/Amazon_ratings.csv')

cnt = ratings.groupby('user_id').count()['rating']
keys = cnt[cnt>3].keys()
ratings = ratings[ratings['user_id'].isin(keys)]

# train, test set 나누기
x = ratings.copy()
y = ratings['user_id']
ratings_train, ratings_test = train_test_split(x, test_size=0.25, stratify=y, random_state=8)

# timestamp 제거
ratings_train = ratings_train[['user_id', 'item_id', 'rating']]
ratings_test = ratings_test[['user_id', 'item_id', 'rating']]

# 사용자 ID와 영화 ID를 연속적인 인덱스로 매핑
user_to_index = {user: idx for idx, user in enumerate(ratings["user_id"].unique())}
item_to_index = {item: idx for idx, item in enumerate(ratings["item_id"].unique())}

# NeuMF는 맵핑된 id를 사용하기 때문에 미리 변환
ratings_train['user_id_conv'] = ratings_train.user_id.map(user_to_index)
ratings_train['item_id_conv'] = ratings_train.item_id.map(item_to_index)
ratings_test['user_id_conv'] = ratings_test.user_id.map(user_to_index)
ratings_test['item_id_conv'] = ratings_test.item_id.map(item_to_index)


# Data already pre-split
train_dataset = CustomDataset(
    user_ids = ratings_train['user_id_conv'].values,
    item_ids = ratings_train['item_id_conv'].values,
    ratings = ratings_train['rating'].values.astype(np.float32)
)

test_dataset = CustomDataset(
    user_ids = ratings_test['user_id_conv'].values,
    item_ids = ratings_test['item_id_conv'].values,
    ratings = ratings_test['rating'].values.astype(np.float32)
)

# Define batch size and other parameters
batch_size = 32
shuffle = True  # Shuffle for training data, typically not for validation data

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create model, loss function, and optimizer
num_users = int(ratings.user_id.nunique())
num_items = int(ratings.item_id.nunique())
num_factors = 220
num_layers = 3

model = NeuMF(num_users, num_items, num_factors, num_layers)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# device configuration
model.to(device)
criterion.to(device)

# Start the training and evaluation process
epochs = 20
patience = 3
train_rmse, val_rmse, best_RMSE = train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs, scheduler, patience)

# NeuMF 결과 확인
print(f'NeuMF Best RMSE: {best_RMSE.item()}')