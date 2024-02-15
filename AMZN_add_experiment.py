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


## 2. Factorization Machine

# Encoding dictionaries
def create_encoding_dict(feature, start_point):
    feature_dict = {}
    for value in set(feature):
        feature_dict[value] = start_point + len(feature_dict)
    return feature_dict, start_point + len(feature_dict)

def encode_data(input, bias, user_dict, item_dict, embeddings_start_idx):
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
        
        # review encoding
        review_embed = ea_case[-384:]        # 해당 리뷰의 임베딩
        for j in range(384):
            x_index.append(embeddings_start_idx+j)
            x_value.append(review_embed[j])

        # target encoding
        data.append([x_index, x_value])
        target.append(ea_case['rating']-bias)

        # 진행 상황 출력
        if (i % 30000) == 0:
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

    def predict_one(self, user_id, movie_id):
        x_idx = np.array([user_dict[user_id], item_dict[movie_id]])
        x_data = np.array([1, 1])
        return self.predict(x_idx, x_data) + w0


## 3. NeuMF

# Dataset 생성
class CustomDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings, text_embeddings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)
        self.text_embeddings = torch.tensor(text_embeddings, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx], self.text_embeddings[idx]

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_layers, embedding_size, num_factors, text_emb_size, dropout=None):
        super().__init__()

        self.dropout = dropout

        # FM part
        self.user_embedding_fm = nn.Embedding(num_users, num_factors)
        self.item_embedding_fm = nn.Embedding(num_items, num_factors)

        # self.w_0 = nn.Parameter(torch.zeros(1))  # global bias
        self.w = nn.Parameter(torch.Tensor(num_factors * 2 + text_emb_size))  # 특성별 가중치
        self.v = nn.Parameter(torch.Tensor(num_factors * 2 + text_emb_size, embedding_size))  # 잠재 요인 가중치

        # MLP part
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size * (2 ** (num_layers - 1)))
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size * (2 ** (num_layers - 1)))

        mlp_input_size = embedding_size * (2 ** (num_layers - 1)) * 2 + text_emb_size

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
        self.final_layer = nn.Linear(222, 1)

        self._init_weight_()

    def _init_weight_(self):
        # Initialize weights here
        nn.init.normal_(self.user_embedding_fm.weight, std=0.01)
        nn.init.normal_(self.item_embedding_fm.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        nn.init.normal_(self.w, std=0.01)
        nn.init.normal_(self.v, std=0.01)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(self.final_layer.weight)
        
        nn.init.kaiming_normal_(self.final_layer.weight)
    
    def forward(self, user_input, item_input, text_input):
        # FM part
        user_emb_mf = self.user_embedding_fm(user_input)
        item_emb_mf = self.item_embedding_fm(item_input)

        x = torch.cat([user_emb_mf, item_emb_mf, text_input], dim=1)

        # 1차 상호작용: 각 사용자와 아이템의 가중치를 곱하고, 그 결과를 모두 더함
        linear_terms = torch.sum(x * self.w, dim=1)


        # 2차 상호작용
        interactions = 0.5 * torch.sum(
            torch.pow(torch.matmul(x, self.v), 2) - torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2)), dim=1)


        # 예측값 계산
        # predict = self.w_0 + linear_terms + interactions
        linear_terms = linear_terms.unsqueeze(-1)
        interactions = interactions.unsqueeze(-1)
        predict = torch.cat([linear_terms, interactions], dim=1)



        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_input)
        item_emb_mlp = self.item_embedding_mlp(item_input)
        mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp, text_input], dim=1)
        mlp_vector = self.mlp_layers(mlp_vector)


        vector = torch.cat([predict, mlp_vector],dim=1)
    

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
        for user, item, rating, text in train_loader:
            user, item, rating, text = user.to(device), item.to(device), rating.float().to(device), text.float().to(device)
            
            optimizer.zero_grad()
            prediction = model(user, item, text)
            loss = criterion(prediction.view(-1), rating.view(-1))
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            
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
        for user, item, rating, text in val_loader:
            user, item, rating, text = user.to(device), item.to(device), rating.float().to(device), text.float().to(device)
            prediction = model(user, item, text)
            loss = criterion(prediction.view(-1), rating.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return torch.sqrt(torch.tensor(avg_loss))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 준비

# 데이터 불러오기
ratings = pd.read_csv('/home/ryu/thesis/data/amazon/Amazon_ratings.csv')
reviews = pd.read_csv('/home/ryu/thesis/data/amazon/Amazon_reviews.csv')

ratings = ratings[['item_id', 'user_id', 'rating']]
reviews = reviews[['item_id', 'user_id', 'text']]

## 기본 전처리
cnt = ratings.groupby('user_id').count()['rating']
keys = cnt[cnt>3].keys()
ratings = ratings[ratings['user_id'].isin(keys)]

with open('/home/ryu/thesis/real_amazon/additional_var/sbert_emb.pickle', 'rb') as f:
    embeddings = pickle.load(f)

emb = pd.DataFrame(embeddings)

data = pd.merge(ratings, reviews, how='left', left_on=['user_id', 'item_id'], right_on=['user_id', 'item_id'])
data = pd.concat([data, emb], axis=1)

## train, test set 나누기
x = data.copy()
y = data['user_id']
ratings_train, ratings_test = train_test_split(x, test_size=0.25, stratify=y, random_state=42)

## Black Sheep 사용자 목록 불러오기
with open('/home/ryu/thesis/real_amazon/black_id.pkl', 'rb') as f:
    black = pickle.load(f)

## train, test 에서 black sheep 사용자만 추출
black_train = ratings_train[ratings_train['user_id'].isin(black)]
black_test = ratings_test[ratings_test['user_id'].isin(black)]
black_all = ratings[ratings['user_id'].isin(black)]

## 사용자 수 구하기 (이후 가중평균 위함)
entire_pop = ratings.user_id.nunique()      # 전체 사용자 수
black_pop = len(black)                      # Black Sheep 사용자 수
rest_pop = entire_pop - black_pop           # 전체 - black sheep = white & gray 사용자 수

## black sheep 제거 데이터
ratings_train = ratings_train[~ratings_train['user_id'].isin(black)]
ratings_test = ratings_test[~ratings_test['user_id'].isin(black)]
ratings = ratings[~ratings['user_id'].isin(black)]

## 사용자 ID와 영화 ID를 연속적인 인덱스로 매핑
user_to_index = {user: idx for idx, user in enumerate(ratings["user_id"].unique())}
item_to_index = {item: idx for idx, item in enumerate(ratings["item_id"].unique())}

## NeuMF는 맵핑된 id를 사용하기 때문에 미리 변환
ratings_train['user_id_conv'] = ratings_train.user_id.map(user_to_index)
ratings_train['item_id_conv'] = ratings_train.item_id.map(item_to_index)
ratings_test['user_id_conv'] = ratings_test.user_id.map(user_to_index)
ratings_test['item_id_conv'] = ratings_test.item_id.map(item_to_index)

## White Sheep 대상 MF 실험 결과 불러오기 (NeuMF 실험 시 사용)
with open('/home/ryu/thesis/real_amazon/additional_var/White_FM/42p_White FM.pkl', 'rb') as f:
    white_results_loaded = pickle.load(f)

## Open saved user_gsu_dict (Gray Sheep id 불러오기)
with open('/home/ryu/thesis/real_amazon/additional_var/1_gsu_data/FM_42_pearson_gsu.pkl', 'rb') as f:
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

blk_bs = {}
blk_weighted_bestseller = {}
blk_weighted_gray_bs = {}
blk_weighted_gray_fm = {}
blk_weighted_neumf = {}

## 2. Black Sheep 실험 (trained with only black_train set; gray sheep 기준에 따라 변함 없기 때문에 미리 계산)
blk_only_bestseller = Biased_Bestseller(black_train, black_test)
print('*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*')
print(f'Black Bestseller RMSE (trained with only black_train): {blk_only_bestseller}')
print('*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*')

## 3. 기준별로 gray sheep id 가져와서 실험


for thresh in gray_dict.keys():

    
    gray_idx = gray_dict[thresh]    # thresh%에 해당하는 Gray sheep 사용자 id 가져오기
    white_rmse = white_results_loaded[thresh]    # thresh%에 해당하는 White Sheep MF 결과 가져오기 (NeuMF 실험 시에만 사용!)

    print('**************************************************')
    print(f'                     {thresh}% 실험 시작                ')
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


    #### 0. Black Sheep Bestseller (trained with White Sheep train set)
    blk_bestseller = Biased_Bestseller(white_train, black_test)

    print(f'{thresh}% Black Bestseller RMSE (trained with white_train): {blk_bestseller}')
    blk_bs[f'{thresh}'] = blk_bestseller


    # #### 1. White Sheep FM ####


    # # 사용자, 아이템, 직업, 성별 인코딩
    # user_dict, start_point = create_encoding_dict(white['user_id'], 0)
    # item_dict, start_point = create_encoding_dict(white['item_id'], start_point)

    # # 텍스트 임베딩
    # text_index = start_point
    # start_point += 384

    # # 전체 특성 수 계산
    # num_x = start_point

    # # train set 평점의 평균값 -> 타겟 변수에서 빼서 평균 평점에 대한 보정 진행
    # white_w0 = np.mean(white_train['rating'])

    # print('Encoding Train Set')
    # w_train_data, w_train_target = encode_data(white_train, white_w0, user_dict, item_dict, text_index)
    # print('Encoding Test Set')
    # w_test_data, w_test_target = encode_data(white_test, white_w0, user_dict, item_dict, text_index)

    # K = 220
    # white_model = FM(num_x, K, w_train_data, w_train_target, w_test_data, w_test_target, alpha=0.0014, beta=0.003,  
    #         iterations=400, tolerance=0.0005, l2_reg=True, verbose=True)

    # result = white_model.test()

    # white_rmse = result[1]


    # print(f'--- {thresh}% white sheep RMSE: {white_rmse} ---')
    # white_fm[f'{thresh}'] = white_rmse

    
    # #### 2. Bestseller with all ratings ####
    # bestseller_rmse = Bestseller(ratings_train, gray_test)

    # print(f'{thresh}% Bestseller RMSE: {bestseller_rmse}')
    # bestseller[f'{thresh}'] = bestseller_rmse

    # # white sheep trained, black sheep tested
    # weight_avg_bs = (blk_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (bestseller_rmse * (len(gray_idx)/entire_pop))
    # print(f'{thresh}% Bestseller weighted RMSE (w/ White Sheep): {weight_avg_bs}')
    # weighted_bestseller[f'{thresh}'] = weight_avg_bs

    # # black sheep trained, black sheep tested
    # blk_weight_avg_bs = (blk_only_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (bestseller_rmse * (len(gray_idx)/entire_pop))
    # print(f'{thresh}% Bestseller weighted RMSE (w/ Black Sheep): {blk_weight_avg_bs}')
    # blk_weighted_bestseller[f'{thresh}'] = blk_weight_avg_bs


    # #### 3. Gray Sheep Only Bestseller ####
    # gsu_bestseller_rmse = Bestseller(gray_train, gray_test)

    # print(f'{thresh}% GSU Bestseller RMSE: {gsu_bestseller_rmse}')
    # gray_bs[f'{thresh}'] = gsu_bestseller_rmse

    # # white sheep trained, black sheep tested
    # weight_avg_gray_bs = (blk_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (gsu_bestseller_rmse * (len(gray_idx)/entire_pop))
    # print(f'{thresh}% GSU Bestseller weighted RMSE (w/ White Sheep): {weight_avg_gray_bs}')
    # weighted_gray_bs[f'{thresh}'] = weight_avg_gray_bs

    # # black sheep trained, black sheep tested
    # blk_weight_avg_gray_bs = (blk_only_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (gsu_bestseller_rmse * (len(gray_idx)/entire_pop))
    # print(f'{thresh}% GSU Bestseller weighted RMSE (w/ Black Sheep): {blk_weight_avg_gray_bs}')
    # blk_weighted_gray_bs[f'{thresh}'] = blk_weight_avg_gray_bs


    # #### 4. Gray Sheep FM ####

    # # 사용자, 아이템, 직업, 성별 인코딩
    # user_dict, start_point = create_encoding_dict(gray['user_id'], 0)
    # item_dict, start_point = create_encoding_dict(gray['item_id'], start_point)

    # # 텍스트 임베딩
    # text_index = start_point
    # start_point += 384

    # # 전체 특성 수 계산
    # num_x = start_point

    # # train set 평점의 평균값 -> 타겟 변수에서 빼서 평균 평점에 대한 보정 진행
    # gray_w0 = np.mean(gray_train['rating'])

    # print('Encoding Train Set')
    # g_train_data, g_train_target = encode_data(gray_train, gray_w0, user_dict, item_dict, text_index)
    # print('Encoding Test Set')
    # g_test_data, g_test_target = encode_data(gray_test, gray_w0, user_dict, item_dict, text_index)

    # K = 220
    # gray_model = FM(num_x, K, g_train_data, g_train_target, g_test_data, g_test_target, alpha=0.0014, beta=0.003,  
    #         iterations=400, tolerance=0.0005, l2_reg=True, verbose=True)

    # gray_result = white_model.test()

    # gsu_fm_rmse = gray_result[1]


    # print(f'{thresh}% Gray Sheep MF RMSE: {gsu_fm_rmse}')
    # gray_fm[f'{thresh}'] = gsu_fm_rmse


    # # white sheep trained, black sheep tested
    # weight_avg_gray_mf = (blk_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (gsu_fm_rmse * (len(gray_idx)/entire_pop))
    # print(f'{thresh}% Gray Sheep MF weighted RMSE (w/ White Sheep): {weight_avg_gray_mf}')
    # weighted_gray_fm[f'{thresh}'] = weight_avg_gray_mf

    # # black sheep trained, black sheep tested
    # blk_weight_avg_gray_mf = (blk_only_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (gsu_fm_rmse * (len(gray_idx)/entire_pop))
    # print(f'{thresh}% Gray Sheep MF weighted RMSE (w/ Black Sheep): {blk_weight_avg_gray_mf}')
    # blk_weighted_gray_fm[f'{thresh}'] = blk_weight_avg_gray_mf



    # #### 5. Gray NeuMF ####

    # 하이퍼파라미터 설정
    num_users = len(set(data.user_id)) + 1
    num_items = len(set(data.item_id)) + 1
    num_layers=3
    embedding_size = 220
    num_factors = 220
    text_emb_size = 384
    epochs = 50
    batch_size = 32
    patience = 3

    # 데이터 로더 정의 
    train_text_embeddings = emb.iloc[ratings_train['user_id_conv']].values
    test_text_embeddings = emb.iloc[ratings_test['user_id_conv']].values

    train_dataset = CustomDataset(
        user_ids = neumf_gray_train['user_id_conv'].values,
        item_ids = neumf_gray_train['item_id_conv'].values,
        ratings = neumf_gray_train['rating'].values.astype(np.float32),
        text_embeddings = train_text_embeddings
    )
    test_dataset = CustomDataset(
        user_ids = neumf_gray_test['user_id_conv'].values,
        item_ids = neumf_gray_test['item_id_conv'].values,
        ratings = neumf_gray_test['rating'].values.astype(np.float32),
        text_embeddings = test_text_embeddings
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델 인스턴스 생성
    model = NeuMF(num_users, num_items, num_layers, embedding_size, num_factors, text_emb_size)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0014, weight_decay=0.015)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)
    criterion.to(device)

    train_rmse, val_rmse, best_RMSE = train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs, scheduler, patience)

    print(f'{thresh}% NeuMF RMSE: {best_RMSE.item()}')
    neumf[f'{thresh}'] = best_RMSE.item()
    

    # white sheep trained, black sheep tested
    weighted_gray_neumf = (blk_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (best_RMSE.item() * (len(gray_idx)/entire_pop))
    print(f'{thresh}% NeuMF weighted RMSE (w/ White Sheep): {weighted_gray_neumf}')
    weighted_neumf[f'{thresh}'] = weighted_gray_neumf

    # black sheep trained, black sheep tested
    blk_weighted_gray_neumf = (blk_only_bestseller * (black_pop/entire_pop)) + (white_rmse * ((rest_pop-len(gray_idx))/entire_pop)) + (best_RMSE.item() * (len(gray_idx)/entire_pop))
    print(f'{thresh}% NeuMF weighted RMSE (w/ Black Sheep): {blk_weighted_gray_neumf}')
    blk_weighted_neumf[f'{thresh}'] = blk_weighted_gray_neumf



    print(f'                {thresh}% 실험 끝                 ')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 실험 결과 저장



with open('/home/ryu/thesis/real_amazon/additional_var/Bestseller/42p_Bestseller Black.pkl', 'wb') as f:
    pickle.dump(blk_bs, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/White_FM/42p_White FM.pkl', 'wb') as f:
#     pickle.dump(white_fm, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/Bestseller/42p_Bestseller.pkl', 'wb') as f:
#     pickle.dump(bestseller, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/Bestseller/42p_Weighted Bestseller.pkl', 'wb') as f:
#     pickle.dump(weighted_bestseller, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/Bestseller/42p_Weighted Bestseller Black.pkl', 'wb') as f:
#     pickle.dump(blk_weighted_bestseller, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/GSU_Bestseller/42p_GSU Bestseller.pkl', 'wb') as f:
#     pickle.dump(gray_bs, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/GSU_Bestseller/42p_Weighted GSU Bestseller.pkl', 'wb') as f:
#     pickle.dump(weighted_gray_bs, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/GSU_Bestseller/42p_Weighted GSU Bestseller Black.pkl', 'wb') as f:
#     pickle.dump(blk_weighted_gray_bs, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/Gray_FM/42p_Gray FM.pkl', 'wb') as f:
#     pickle.dump(gray_fm, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/Gray_FM/42p_Weighted Gray FM.pkl', 'wb') as f:
#     pickle.dump(weighted_gray_fm, f)
# with open('/home/ryu/thesis/real_amazon/additional_var/Gray_FM/42p_Weighted Gray FM Black.pkl', 'wb') as f:
#     pickle.dump(blk_weighted_gray_fm, f)
with open('/home/ryu/thesis/real_amazon/additional_var/Gray_NeuMF/42p_NeuMF.pkl', 'wb') as f:
    pickle.dump(neumf, f)
with open('/home/ryu/thesis/real_amazon/additional_var/Gray_NeuMF/42p_Weighted NeuMF.pkl', 'wb') as f:
    pickle.dump(weighted_neumf, f)
with open('/home/ryu/thesis/real_amazon/additional_var/Gray_NeuMF/42p_Weighted NeuMF Black.pkl', 'wb') as f:
    pickle.dump(blk_weighted_neumf, f)


print(f'               실험 결과 저장 완료 😎               ')


