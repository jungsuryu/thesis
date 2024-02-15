
# 필요한 모듈 불러오기

import os

import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action = 'ignore')

import pickle
import itertools

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

## 2. Factorization Machine

# Encoding dictionaries
def create_encoding_dict(feature, start_point):
    feature_dict = {}
    for value in set(feature):
        feature_dict[value] = start_point + len(feature_dict)
    return feature_dict, start_point + len(feature_dict)

def encode_data(input, bias, user_dict, item_dict, occ_dict, gender_dict, genre_dict, age_index, genres):
    data = []
    target = []

    for i in range(len(input)):
        ea_case = input.iloc[i]
        x_index = []
        x_value = []

        # user id encoding
        x_index.append(user_dict[ea_case['user_id']])
        x_value.append(1.)

        # item id encoding
        x_index.append(item_dict[ea_case['item_id']])
        x_value.append(1.)

        # occupation id encoding
        x_index.append(occ_dict[ea_case['occupation']])
        x_value.append(1.)

        # gender id encoding
        x_index.append(gender_dict[ea_case['sex']])
        x_value.append(1.)

        # genre encoding
        for j in genres:
            if ea_case[j] == 1:
                x_index.append(genre_dict[j])
                x_value.append(1.)
        
        # age encoding
        x_index.append(age_index)
        x_value.append(ea_case['age'])

        # target encoding
        data.append([x_index, x_value])
        target.append(ea_case['rating']-bias)

        # 진행 상황 출력
        if (i % 100000) == 0:
            print('Encoding ', i, 'cases...')
    
    return data, target

class FM():
    def __init__(self, N, K, train_x, train_y, test_x, test_y, alpha, beta, iterations=100, tolerance=0.005, l2_reg=True, verbose=True): # 초기화
        self.K = K                          # Number of latent factors
        self.N = N                          # Number of x (variables)
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.l2_reg = l2_reg
        self.tolerance = tolerance
        self.verbose = verbose

        # w와 v 초기화
        self.w = np.random.normal(scale=1./self.N, size=(self.N)) # 사이즈는 변수의 수만큼. 변수마다 bias 하나
        self.v = np.random.normal(scale=1./self.K, size=(self.N, self.K)) # 변수의 수 * K

        # Train/Test 분리
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

    def test(self):                                     # Training 하면서 RMSE 계산 
        # SGD를 iterations 숫자만큼 수행
        best_RMSE = float('inf') # stop 위해
        best_iteration = 0
        training_process = []
        for i in range(self.iterations): # 600번
            rmse1 = self.sgd(self.train_x, self.train_y)        # SGD & Train RMSE 계산
            rmse2 = self.test_rmse(self.test_x, self.test_y)    # Test RMSE 계산     
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.6f ; Test RMSE = %.6f" % (i+1, rmse1, rmse2))
            if best_RMSE > rmse2:                       # New best record
                best_RMSE = rmse2
                best_iteration = i
            elif (rmse2 - best_RMSE) > self.tolerance:  # RMSE is increasing over tolerance
                break
        print(best_iteration, best_RMSE)
        return training_process, best_RMSE
        
    # w, v 업데이트를 위한 Stochastic gradient descent 
    def sgd(self, x_data, y_data):
        y_pred = []
        for data, y in zip(x_data, y_data): # 100,000번. x_data, y_data가 100,000개
            x_idx = data[0] # 데이터의 첫번째 (x_index, x_value)에 대한 인덱스 받아옴
            x_0 = np.array(data[1])     # xi axis=0 [1, 2, 3] (1차원)
            x_1 = x_0.reshape(-1, 1)    # xi axis=1 [[1], [2], [3]] (2차원: V matrix와 계산 위해서)
    
            # biases
            bias_score = np.sum(self.w[x_idx] * x_0) # 여기선 x_0를 1차원으로 사용. w matrix는 1차원이기 때문
    
            # score 계산
            vx = self.v[x_idx] * (x_1)          # v matrix * x (브로드캐스팅)
            sum_vx = np.sum(vx, axis=0)         # sigma(vx): 칼럼으로 쭉 더한 것 (element K개 (=350개))
            sum_vx_2 = np.sum(vx * vx, axis=0)  # ( v matrix * x )의 제곱: element 350개
            latent_score = 0.5 * np.sum(np.square(sum_vx) - sum_vx_2)

            # 예측값 계산
            y_hat = bias_score + latent_score # bias까지 더하면 최종 예측값 (전체 평균은 전에 뺐기 때문에 따로 또 빼주지 않음)
            y_pred.append(y_hat) # y_pred 75,000개 (아까 train,test 분리함)
            error = y - y_hat # 에러 구했으니까 아래에서 업데이트 가능
            # w, v 업데이트 (week 7 수업자료에 있는 update rule)
            if self.l2_reg:     # regularization이 있는 경우
                self.w[x_idx] += error * self.alpha * (x_0 - self.beta * self.w[x_idx])
                self.v[x_idx] += error * self.alpha * ((x_1) * sum(vx) - (vx * x_1) - self.beta * self.v[x_idx])
            else:               # regularization이 없는 경우
                self.w[x_idx] += error * self.alpha * x_0
                self.v[x_idx] += error * self.alpha * ((x_1) * sum(vx) - (vx * x_1))
        return RMSE(y_data, y_pred) 

    def test_rmse(self, x_data, y_data): # test set에 대한 RMSE
        y_pred = []
        for data , y in zip(x_data, y_data):
            y_hat = self.predict(data[0], data[1])
            y_pred.append(y_hat)
        return RMSE(y_data, y_pred)

    def predict(self, idx, x):
        x_0 = np.array(x)
        x_1 = x_0.reshape(-1, 1)

        # biases
        bias_score = np.sum(self.w[idx] * x_0)

        # score 계산
        vx = self.v[idx] * (x_1)
        sum_vx = np.sum(vx, axis=0)
        sum_vx_2 = np.sum(vx * vx, axis=0)
        latent_score = 0.5 * np.sum(np.square(sum_vx) - sum_vx_2)

        # 예측값 계산
        y_hat = bias_score + latent_score
        return y_hat
    
    def predict_one(self, user_id, item_id):
        x_idx = np.array([user_dict[user_id], item_dict[item_id]])
        x_data = np.array([1, 1])
        return self.predict(x_idx, x_data) + w0
        

## 3. NeuMF

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
    def __init__(self, patience=5, min_delta=0.005):
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

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 준비

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


# 사용자 ID와 영화 ID를 연속적인 인덱스로 매핑
user_to_index = {user: idx for idx, user in enumerate(data['user_id'].unique())}
item_to_index = {item: idx for idx, item in enumerate(data['item_id'].unique())}
occ_to_index = {occ: idx for idx, occ in enumerate(data['occupation'].unique())}
sex_to_index=  {gender: idx for idx, gender in enumerate(data['sex'].unique())}

# NeuMF는 맵핑된 id를 사용하기 때문에 미리 변환
ratings_train['user_id_conv'] = ratings_train.user_id.map(user_to_index)
ratings_train['item_id_conv'] = ratings_train.item_id.map(item_to_index)
ratings_train['occ_id_conv'] = ratings_train.occupation.map(occ_to_index)
ratings_train['sex_id_conv'] = ratings_train.sex.map(sex_to_index)

ratings_test['user_id_conv'] = ratings_test.user_id.map(user_to_index)
ratings_test['item_id_conv'] = ratings_test.item_id.map(item_to_index)
ratings_test['occ_id_conv'] = ratings_test.occupation.map(occ_to_index)
ratings_test['sex_id_conv'] = ratings_test.sex.map(sex_to_index)

# # White Sheep 대상 MF 실험 결과 불러오기 (NeuMF 실험 시 사용)
with open('/home/ryu/thesis/real_movielens/additional_var/White_FM/8c_White FM.pkl', 'rb') as f:
    white_results_loaded = pickle.load(f)

## Open saved user_gsu_dict (Gray Sheep id 불러오기)
with open('/home/ryu/thesis/real_movielens/additional_var/1_gsu_data/FM_8_cosine_gsu.pkl', 'rb') as f:
    gray_dict = pickle.load(f)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 시작

## 1. 결과 저장할 딕셔너리 생성

white_fm = {}
bestseller = {}
weighted_bestseller = {}
gray_bs = {}
weighted_gray_bs = {}
gray_fm = {}
weighted_gray_fm = {}
neumf = {}
weighted_neumf = {}



## 2. 기준별로 gray sheep id 가져와서 실험

# test_list = [str(n) for n in range(10, 51, 5)]
# th = list(range(11, 20)) + list(range(21, 30)) + list(range(31, 40)) + list(range(41, 50))
# test_list = [str(n) for n in th]

for thresh in gray_dict.keys():

    
    gray_idx = gray_dict[thresh]    # thresh%에 해당하는 Gray sheep 사용자 id 가져오기
    white_rmse = white_results_loaded[thresh]    # thresh%에 해당하는 White Sheep MF 결과 가져오기 (NeuMF 실험 시에만 사용!)

    print('**************************************************')
    print(f'                {thresh}% 실험 시작                ')
    print('**************************************************')


    # white, gray sheep 사용자 분리
    white = data[~data['user_id'].isin(gray_idx)]
    gray = data[data['user_id'].isin(gray_idx)]

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
    

    # #### 1. White Sheep FM ####

    # ## 사용자, 아이템, 직업, 성별 인코딩
    # user_dict, start_point = create_encoding_dict(white['user_id'], 0)
    # item_dict, start_point = create_encoding_dict(white['item_id'], start_point)
    # occ_dict, start_point = create_encoding_dict(white['occupation'], start_point)
    # gender_dict, start_point = create_encoding_dict(white['sex'], start_point)

    # ## 장르 인코딩
    # all_genres = [x.split('|') for x in movies['genre'].values]
    # genres = list(set(list(itertools.chain(*all_genres))))

    # genre_dict, start_point = create_encoding_dict(genres, start_point)

    # # 연령 인덱싱 (연령은 하나의 독립적인 특성으로 취급하기 위해 +1 (단일 차원))
    # age_index = start_point
    # start_point += 1

    # # 전체 특성 수 계산
    # num_x = start_point

    # # train set 평점의 평균값 -> 타겟 변수에서 빼서 평균 평점에 대한 보정 진행
    # white_w0 = np.mean(white_train['rating'])

    # print('Encoding Train Set')
    # w_train_data, w_train_target = encode_data(white_train, white_w0, user_dict, item_dict, occ_dict, gender_dict, genre_dict, age_index, genres)
    # print('Encoding Test Set')
    # w_test_data, w_test_target = encode_data(white_test, white_w0, user_dict, item_dict, occ_dict, gender_dict, genre_dict, age_index, genres)

    # K = 220
    # white_model = FM(num_x, K, w_train_data, w_train_target, w_test_data, w_test_target, alpha=0.0024, beta=0.05,  
    #         iterations=400, tolerance=0.0005, l2_reg=True, verbose=True)

    # result = white_model.test()

    # white_rmse = result[1]

    # print(f'--- {thresh}% white sheep RMSE: {white_rmse} ---')
    # white_fm[f'{thresh}'] = white_rmse
    
    # #### 2. Bestseller with all ratings ####
    # bestseller_rmse = Bestseller(ratings_train, gray_test)

    # print(f'{thresh}% Bestseller RMSE: {bestseller_rmse}')
    # bestseller[f'{thresh}'] = bestseller_rmse

    # weight_avg_bs = (white_rmse * (1 - (int(thresh)*0.01))) + (bestseller_rmse * (int(thresh)*0.01))
    # print(f'{thresh}% Bestseller weighted RMSE: {weight_avg_bs}')
    # weighted_bestseller[f'{thresh}'] = weight_avg_bs

    # #### 3. Gray Sheep Only Bestseller ####
    # gsu_bestseller_rmse = Bestseller(gray_train, gray_test)

    # print(f'{thresh}% GSU Bestseller RMSE: {gsu_bestseller_rmse}')
    # gray_bs[f'{thresh}'] = gsu_bestseller_rmse

    # weight_avg_gray_bs = (white_rmse * (1 - (int(thresh)*0.01))) + (gsu_bestseller_rmse * (int(thresh)*0.01))
    # print(f'{thresh}% GSU Bestseller weighted RMSE: {weight_avg_gray_bs}')
    # weighted_gray_bs[f'{thresh}'] = weight_avg_gray_bs

    # #### 4. Gray Sheep FM ####

    # ## 사용자, 아이템, 직업, 성별 인코딩
    # user_dict, start_point = create_encoding_dict(gray['user_id'], 0)
    # item_dict, start_point = create_encoding_dict(gray['item_id'], start_point)
    # occ_dict, start_point = create_encoding_dict(gray['occupation'], start_point)
    # gender_dict, start_point = create_encoding_dict(gray['sex'], start_point)

    # ## 장르 인코딩
    # all_genres = [x.split('|') for x in movies['genre'].values]
    # genres = list(set(list(itertools.chain(*all_genres))))

    # genre_dict, start_point = create_encoding_dict(genres, start_point)

    # # 연령 인덱싱 (연령은 하나의 독립적인 특성으로 취급하기 위해 +1 (단일 차원))
    # age_index = start_point
    # start_point += 1

    # # 전체 특성 수 계산
    # num_x = start_point

    # # train set 평점의 평균값 -> 타겟 변수에서 빼서 평균 평점에 대한 보정 진행
    # gray_w0 = np.mean(gray_train['rating'])

    # print('Encoding Train Set')
    # g_train_data, g_train_target = encode_data(gray_train, gray_w0, user_dict, item_dict, occ_dict, gender_dict, genre_dict, age_index, genres)
    # print('Encoding Test Set')
    # g_test_data, g_test_target = encode_data(gray_test, gray_w0, user_dict, item_dict, occ_dict, gender_dict, genre_dict, age_index, genres)

    # K = 220
    # gray_model = FM(num_x, K, g_train_data, g_train_target, g_test_data, g_test_target, alpha=0.0024, beta=0.05,  
    #         iterations=400, tolerance=0.0005, l2_reg=True, verbose=True)
    
    # gray_result = gray_model.test()

    # gsu_fm_rmse = gray_result[1]

    # print(f'{thresh}% Gray Sheep FM RMSE: {gsu_fm_rmse}')
    # gray_fm[f'{thresh}'] = gsu_fm_rmse

    # weight_avg_gray_fm = (white_rmse * (1 - (int(thresh)*0.01))) + (gsu_fm_rmse * (int(thresh)*0.01))
    # print(f'{thresh}% GSU FM: {weight_avg_gray_fm}')
    # weighted_gray_fm[f'{thresh}'] = weight_avg_gray_fm


    #### 5. Gray NeuMF ####
    # 데이터 로더 정의 
    train_dataset = CustomDataset(
        user_ids = neumf_gray_train['user_id_conv'].values,
        item_ids = neumf_gray_train['item_id_conv'].values,
        ratings = neumf_gray_train['rating'].values.astype(np.float32),
        occupation = neumf_gray_train['occ_id_conv'].values,
        gender = neumf_gray_train['sex_id_conv'].values,
        genre = neumf_gray_train.iloc[:, 6:24].values,
        age = neumf_gray_train['age'].values
    )

    test_dataset = CustomDataset(
        user_ids = neumf_gray_test['user_id_conv'].values,
        item_ids = neumf_gray_test['item_id_conv'].values,
        ratings = neumf_gray_test['rating'].values.astype(np.float32),
        occupation = neumf_gray_test['occ_id_conv'].values,
        gender = neumf_gray_test['sex_id_conv'].values,
        genre = neumf_gray_test.iloc[:, 6:24].values,
        age = neumf_gray_test['age'].values
    )

    # Create model, loss function, and optimizer
    # 하이퍼파라미터 설정
    num_users = len(set(data.user_id)) + 1
    num_items = len(set(data.item_id)) + 1
    num_layers=3
    embedding_size = 220
    num_factors = 220
    num_occ = len(set(data.occupation)) + 1
    epochs = 100
    batch_size = 1024
    patience = 3

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

    # device configuration
    model.to(device)
    criterion.to(device)

    # Start the training and evaluation process

    train_rmse, val_rmse, best_RMSE = train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs, scheduler, patience)


    print(f'{thresh}% NeuMF RMSE: {best_RMSE.item()}')
    neumf[f'{thresh}'] = best_RMSE.item()
    
    weighted_gray_neumf = (white_rmse * (1 - (int(thresh)*0.01))) + (best_RMSE.item() * (int(thresh)*0.01))
    print(f'NeuMF weighted Average: {weighted_gray_neumf}')
    weighted_neumf[f'{thresh}'] = weighted_gray_neumf



    print(f'                {thresh}% 실험 끝                 ')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 결과 저장

# with open('/home/ryu/thesis/real_movielens/additional_var/White_FM/8c_White FM.pkl', 'wb') as f:
#     pickle.dump(white_fm, f)
# with open('/home/ryu/thesis/real_movielens/additional_var/Bestseller/8c_Bestseller.pkl', 'wb') as f:
#     pickle.dump(bestseller, f)
# with open('/home/ryu/thesis/real_movielens/additional_var/Bestseller/8c_Weighted Bestseller.pkl', 'wb') as f:
#     pickle.dump(weighted_bestseller, f)
# with open('/home/ryu/thesis/real_movielens/additional_var/GSU_Bestseller/8c_GSU Bestseller.pkl', 'wb') as f:
#     pickle.dump(gray_bs, f)
# with open('/home/ryu/thesis/real_movielens/additional_var/GSU_Bestseller/8c_Weighted GSU Bestseller.pkl', 'wb') as f:
#     pickle.dump(weighted_gray_bs, f)
# with open('/home/ryu/thesis/real_movielens/additional_var/Gray_FM/8c_Gray FM.pkl', 'wb') as f:
#     pickle.dump(gray_fm, f)
# with open('/home/ryu/thesis/real_movielens/additional_var/Gray_FM/8c_Weighted Gray FM.pkl', 'wb') as f:
#     pickle.dump(weighted_gray_fm, f)
with open('/home/ryu/thesis/real_movielens/additional_var/Gray_NeuMF/8c_NeuMF.pkl', 'wb') as f:
    pickle.dump(neumf, f)
with open('/home/ryu/thesis/real_movielens/additional_var/Gray_NeuMF/8c_Weighted NeuMF.pkl', 'wb') as f:
    pickle.dump(weighted_neumf, f)


print(f'               실험 결과 저장 완료 😎               ')


