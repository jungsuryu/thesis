
# 필요한 모듈 불러오기

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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# RMSE 준비

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


# 모델 준비

## 1. Bestseller
def Biased_Bestseller(train_data, test_data):

    train = train_data.copy()
    test = test_data.copy()

    # 아이템별 평균 평점 계산
    rating_mean = train.groupby('item_id')['rating'].mean()
    test = test.join(rating_mean, on='item_id', rsuffix='_item')

    # 전체 평균 평점 계산
    global_mean = train['rating'].mean()
    test['rating_item'].fillna(train['rating'].mean(), inplace=True)

    # 사용자별 평균 평점
    user_mean = train.groupby('user_id')['rating'].mean()
    test = test.join(user_mean, on='user_id', rsuffix='_user')
    
    
    test['predicted_rating'] = test['rating_item'] - global_mean + test['rating_user']

    rmse_result = RMSE(test['rating'], test['predicted_rating'])

    return rmse_result

## 2. Matrix Factorization

class MF_base():
    ##### CLASS INITIALIZATION AND INDEXING ######
    def __init__(self, ratings, K, alpha, beta, iterations, tolerance=0.005, verbose=True):
        """
        Initialize the MF_base object.
        
        :param ratings: DataFrame, user-item interaction data
        :param K: int, number of latent features
        :param alpha: float, learning rate
        :param beta: float, regularization parameter
        :param iterations: int, number of iterations for stochastic gradient descent
        :param tolerance: float, early stopping tolerance for RMSE increase
        :param verbose: bool, whether to print progress messages
        """
        # Convert the user-item matrix to a numpy array
        self.R = np.array(ratings)
        # Map user and item IDs to indices
        self.user_id_index = {user_id: i for i, user_id in enumerate(ratings.index)}
        self.item_id_index = {item_id: i for i, item_id in enumerate(ratings.columns)}
        self.index_user_id = {i: user_id for user_id, i in self.user_id_index.items()}
        self.index_item_id = {i: item_id for item_id, i in self.item_id_index.items()}
        # initialize other variables
        self.num_users, self.num_items = self.R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Placeholder for the test set
        self.test_set = None
    
    ##### TESTING AND RMSE CALCULATION #####
    def set_test(self, ratings_test):
        """
        Set the test dataset and update the user-item matrix to exclude test ratings.
        
        :param ratings_test: DataFrame, the testing data
        :return: list, the test dataset with user and item indices and ratings
        """
        test_set = []
        for _, row in ratings_test.iterrows():
            user_idx = self.user_id_index.get(row['user_id'], None)
            item_idx = self.item_id_index.get(row['item_id'], None)
            
            if user_idx is not None and item_idx is not None:
                test_set.append([user_idx, item_idx, row['rating']])
                self.R[user_idx, item_idx] = 0  # Set the rating to 0 in the training data
        
        self.test_set = test_set
        return test_set

    def train(self, allow_increase=5):
        """
        Train the model using stochastic gradient descent and calculate train and test RMSE.
        
        :param allow_increase: int, allowed number of iterations with RMSE increase
        :return: list, training process with iteration, train RMSE, and test RMSE
        """
        # Initialize user-feature and item-feature matrices
        ### set the standard deviation to 1/self.K for faster and more stable convergence
        ### Gaussian distribution
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize bias terms
        self.b_u = np.zeros(self.num_users)        # user bias
        self.b_d = np.zeros(self.num_items)        # item bias
        self.b = np.mean(self.R[self.R.nonzero()]) # 모든 아이템 평점 평균

        # List of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]

        # Stochastic gradient descent
        best_RMSE = float('inf')
        best_iteration = 0
        training_process = []
        increase_count = 0
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            train_rmse = self.rmse()
            test_rmse = self.test_rmse()
            training_process.append((i, train_rmse, test_rmse))
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Iteration: {i + 1} ; Train RMSE: {train_rmse:.6f} ; Test RMSE: {test_rmse:.6f}")
                
            if test_rmse < best_RMSE:
                best_RMSE = test_rmse
                best_iteration = i
                increase_count = 0  # Reset the increase count
            elif (test_rmse - best_RMSE) > self.tolerance:
                increase_count += 1
                if increase_count > allow_increase:
                    break  # Stop if RMSE increases for more than 'allow_increase' consecutive iterations

        print(f"Best Iteration: {best_iteration} ; Best Test RMSE: {best_RMSE:.6f}")
        return best_iteration, best_RMSE
    
    def sgd(self):
        """
        Perform stochastic gradient descent.
        """
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)
            
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])
            
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])
            
    def rmse(self):
        """
        Calculate root mean square error on the training data.
        
        :return: float, the RMSE value
        """
        predictions = self.R.copy()
        rows, cols = self.R.nonzero()
        for i, j in zip(rows, cols):
            predictions[i, j] = self.get_prediction(i, j)
            
        errors = [(self.R[i, j] - predictions[i, j]) for i, j in zip(rows, cols) if self.R[i, j] > 0]
        return np.sqrt(np.mean(np.array(errors) ** 2))
    
    def test_rmse(self):
        """
        Calculate root mean square error on the test data.
        
        :return: float, the RMSE value
        """
        error = 0
        for user_idx, item_idx, rating in self.test_set:
            predicted = self.get_prediction(user_idx, item_idx)
            error += (rating - predicted) ** 2
        return np.sqrt(error / len(self.test_set))

    ##### PREDICTION #####
    def get_prediction(self, i, j):
        """
        Get the predicted rating for a given user and item index.
        
        :param i: int, user index
        :param j: int, item index
        :return: float, predicted rating
        """
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def get_one_prediction(self, user_id, item_id):
        """
        Get the predicted rating for a given user ID and item ID.
        
        :param user_id: str or int, user ID
        :param item_id: str or int, item ID
        :return: float or str, predicted rating or error message
        """
        user_idx = self.user_id_index.get(user_id, None)
        item_idx = self.item_id_index.get(item_id, None)
        
        if user_idx is not None and item_idx is not None:
            return self.get_prediction(user_idx, item_idx)
        else:
            return "User or Item ID not found in training data"
        

## 3. NeuMF

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


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 준비

## 데이터 불러오기
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
movie_cols = ['item_id', 'title', 'genre'] 

ratings = pd.read_csv('/home/ryu/thesis/data/ml-1m/ratings.dat', sep='::', names=ratings_cols, engine='python', encoding = "ISO-8859-1")
movies = pd.read_csv('/home/ryu/thesis/data/ml-1m/movies.dat', sep='::', names=movie_cols, engine='python', encoding = "ISO-8859-1")

## 평점 없는 영화 제거
movies_in_rating = ratings['item_id'].unique()
movies = movies[movies['item_id'].isin(movies_in_rating)]

## train, test set 나누기
x = ratings.copy()
y = ratings['user_id']
ratings_train, ratings_test = train_test_split(x, test_size=0.25, stratify=y, random_state=8)

## timestamp 제거
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

# White Sheep 대상 MF 실험 결과 불러오기 (NeuMF 실험 시 사용)
with open('8c_White MF.pkl', 'rb') as f:
    white_results_loaded = pickle.load(f)

## Open saved user_gsu_dict (Gray Sheep id 불러오기)
with open('8_cosine_gsu.pkl', 'rb') as f:
    gray_dict = pickle.load(f)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 시작

## 1. 결과 저장할 딕셔너리 생성

white_mf = {}
bestseller = {}
weighted_bestseller = {}
gray_bs = {}
weighted_gray_bs = {}
gray_mf = {}
weighted_gray_mf = {}
neumf = {}
weighted_neumf = {}

## 2. 기준별로 gray sheep id 가져와서 실험

for thresh in gray_dict.keys():

    
    gray_idx = gray_dict[thresh]    # thresh%에 해당하는 Gray sheep 사용자 id 가져오기
    white_rmse = white_results_loaded[thresh]    # thresh%에 해당하는 White Sheep MF 결과 가져오기 (NeuMF 실험 시에만 사용!)

    print('**************************************************')
    print(f'                {thresh}% 실험 시작                ')
    print('**************************************************')


    # white, gray sheep 사용자 분리
    white = ratings[~ratings['user_id'].isin(gray_idx)]
    gray = ratings[ratings['user_id'].isin(gray_idx)]

    white_train = ratings_train[~ratings_train['user_id'].isin(gray_idx)]
    white_test = ratings_test[~ratings_test['user_id'].isin(gray_idx)]

    gray_train = ratings_train[ratings_train['user_id'].isin(gray_idx)]
    gray_test = ratings_test[ratings_test['user_id'].isin(gray_idx)]

    # NeuMF 데이터 준비
    gray_new_idx = []
    for g in gray_idx:
        gray_new_idx.append(user_to_index[g])

    neumf_gray_train = ratings_train[ratings_train['user_id_conv'].isin(gray_new_idx)]
    neumf_gray_test = ratings_test[ratings_test['user_id_conv'].isin(gray_new_idx)]
    

    #### 1. White Sheep MF ####
    R_white = white.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    white_model = MF_base(R_white, K=220, alpha=0.0024, beta=0.05, iterations=400, tolerance=0.0005, verbose=True)
    white_test_set = white_model.set_test(white_test)
    white_mf_result = white_model.train()

    white_rmse = white_mf_result[1]

    print(f'--- {thresh}% white sheep RMSE: {white_rmse} ---')
    white_mf[f'{thresh}'] = white_rmse
    
    #### 2. Bestseller with all ratings ####
    bestseller_rmse = Biased_Bestseller(ratings_train, gray_test)

    print(f'{thresh}% Bestseller RMSE: {bestseller_rmse}')
    bestseller[f'{thresh}'] = bestseller_rmse

    weight_avg_bs = (white_rmse * (1 - (int(thresh)*0.01))) + (bestseller_rmse * (int(thresh)*0.01))
    print(f'{thresh}% Bestseller weighted RMSE: {weight_avg_bs}')
    weighted_bestseller[f'{thresh}'] = weight_avg_bs

    #### 3. Gray Sheep Only Bestseller ####
    gsu_bestseller_rmse = Biased_Bestseller(gray_train, gray_test)

    print(f'{thresh}% GSU Bestseller RMSE: {gsu_bestseller_rmse}')
    gray_bs[f'{thresh}'] = gsu_bestseller_rmse

    weight_avg_gray_bs = (white_rmse * (1 - (int(thresh)*0.01))) + (gsu_bestseller_rmse * (int(thresh)*0.01))
    print(f'{thresh}% GSU Bestseller weighted RMSE: {weight_avg_gray_bs}')
    weighted_gray_bs[f'{thresh}'] = weight_avg_gray_bs

    #### 4. Gray Sheep MF ####
    R_gray = gray.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    gsu_model = MF_base(R_gray, K=220, alpha=0.0024, beta=0.05, iterations=400, tolerance=0.0005, verbose=True)
    gsu_test_set = gsu_model.set_test(gray_test)
    gsu_mf_result = gsu_model.train()

    gsu_mf_rmse = gsu_mf_result[1]

    print(f'{thresh}% Gray Sheep MF RMSE: {gsu_mf_rmse}')
    gray_mf[f'{thresh}'] = gsu_mf_rmse

    weight_avg_gray_mf = (white_rmse * (1 - (int(thresh)*0.01))) + (gsu_mf_rmse * (int(thresh)*0.01))
    print(f'{thresh}% GSU MF: {weight_avg_gray_mf}')
    weighted_gray_mf[f'{thresh}'] = weight_avg_gray_mf


    #### 5. Gray NeuMF ####
    # Data already pre-split
    train_dataset = CustomDataset(
        user_ids = neumf_gray_train['user_id_conv'].values,
        item_ids = neumf_gray_train['item_id_conv'].values,
        ratings = neumf_gray_train['rating'].values.astype(np.float32)
    )

    test_dataset = CustomDataset(
        user_ids = neumf_gray_test['user_id_conv'].values,
        item_ids = neumf_gray_test['item_id_conv'].values,
        ratings = neumf_gray_test['rating'].values.astype(np.float32)
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Example scheduler

    # device configuration
    model.to(device)
    criterion.to(device)

    # Start the training and evaluation process
    epochs = 20
    patience = 3
    train_rmse, val_rmse, best_RMSE = train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs, scheduler, patience)


    print(f'{thresh}% NeuMF RMSE: {best_RMSE.item()}')
    neumf[f'{thresh}'] = best_RMSE.item()
    
    weighted_gray_neumf = (white_rmse * (1 - (int(thresh)*0.01))) + (best_RMSE.item() * (int(thresh)*0.01))
    print(f'NeuMF weighted Average: {weighted_gray_neumf}')
    weighted_neumf[f'{thresh}'] = weighted_gray_neumf



print(f'                {thresh}% 실험 끝                 ')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 결과 저장

with open('/home/ryu/thesis/real_movielens/basic_var/White_MF/8c_White MF.pkl', 'wb') as f:
    pickle.dump(white_mf, f)
with open('/home/ryu/thesis/real_movielens/basic_var/Bestseller/8c_Bestseller.pkl', 'wb') as f:
    pickle.dump(bestseller, f)
with open('/home/ryu/thesis/real_movielens/basic_var/Bestseller/8c_Weighted Bestseller.pkl', 'wb') as f:
    pickle.dump(weighted_bestseller, f)
with open('/home/ryu/thesis/real_movielens/basic_var/GSU_Bestseller/8c_GSU Bestseller.pkl', 'wb') as f:
    pickle.dump(gray_bs, f)
with open('/home/ryu/thesis/real_movielens/basic_var/GSU_Bestseller/8c_Weighted GSU Bestseller.pkl', 'wb') as f:
    pickle.dump(weighted_gray_bs, f)
with open('/home/ryu/thesis/real_movielens/basic_var/Gray_MF/8c_Gray MF.pkl', 'wb') as f:
    pickle.dump(gray_mf, f)
with open('/home/ryu/thesis/real_movielens/basic_var/Gray_MF/8c_Weighted Gray MF.pkl', 'wb') as f:
    pickle.dump(weighted_gray_mf, f)
with open('/home/ryu/thesis/real_movielens/basic_var/Gray_NeuMF/8c_NeuMF.pkl', 'wb') as f:
    pickle.dump(neumf, f)
with open('/home/ryu/thesis/real_movielens/basic_var/Gray_NeuMF/8c_Weighted NeuMF.pkl', 'wb') as f:
    pickle.dump(weighted_neumf, f)


print(f'               실험 결과 저장 완료 😎               ')


