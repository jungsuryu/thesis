import os
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action = 'ignore')

import pickle
import itertools

# ------------------------------------------------------------------------------------------------------------------------------

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

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
        x_index.append(item_dict[ea_case['movie_id']])
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
        return training_process
        
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

# ------------------------------------------------------------------------------------------------------------------------------


# 데이터 불러오기
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
movie_cols = ['movie_id', 'title', 'genre']
user_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']

ratings = pd.read_csv('/home/ryu/thesis/data/ml-1m/ratings.dat', sep='::', names=ratings_cols, engine='python', encoding = "ISO-8859-1")
movies = pd.read_csv('/home/ryu/thesis/data/ml-1m/movies.dat', sep='::', names=movie_cols, engine='python', encoding = "ISO-8859-1")
users = pd.read_csv('/home/ryu/thesis/data/ml-1m/users.dat', sep='::', names=user_cols, engine='python', encoding = "ISO-8859-1")

# 평점 없는 영화 제거
movies_in_rating = ratings['movie_id'].unique()
movies = movies[movies['movie_id'].isin(movies_in_rating)]

# 장르 정리
genres_df = movies['genre'].str.get_dummies(sep='|')
movies = pd.concat([movies, genres_df], axis=1)
# movies = movies.drop(['genre'], axis=1)

# 필요한 정보만 추출
users = users[['user_id', 'age', 'sex', 'occupation']]

# 사용자, 아이템, 직업, 성별 인코딩
user_dict, start_point = create_encoding_dict(users['user_id'], 0)
item_dict, start_point = create_encoding_dict(movies['movie_id'], start_point)
occ_dict, start_point = create_encoding_dict(users['occupation'], start_point)
gender_dict, start_point = create_encoding_dict(users['sex'], start_point)

# 장르 인코딩
all_genres = [x.split('|') for x in movies['genre'].values]
genres = list(set(list(itertools.chain(*all_genres))))

genre_dict, start_point = create_encoding_dict(genres, start_point)

# 연령 인덱싱 (연령은 하나의 독립적인 특성으로 취급하기 위해 +1 (단일 차원))
age_index = start_point
start_point += 1

# 전체 특성 수 계산
num_x = start_point

# 데이터 합병
data = pd.merge(ratings, movies, how='inner', on='movie_id')
data = pd.merge(data, users, how='inner', on='user_id')
data['age'] /= 50

x = data.copy()
y = data['user_id']
ratings_train, ratings_test = train_test_split(x, test_size=0.25, stratify=y, random_state=8)

# train set 평점의 평균값 -> 타겟 변수에서 빼서 평균 평점에 대한 보정 진행
w0 = np.mean(ratings_train['rating'])

print('Encoding Train Set')
train_data, train_target = encode_data(ratings_train, w0, user_dict, item_dict, occ_dict, gender_dict, genre_dict, age_index, genres)
print('Encoding Test Set')
test_data, test_target = encode_data(ratings_test, w0, user_dict, item_dict, occ_dict, gender_dict, genre_dict, age_index, genres)


K = 220
model = FM(num_x, K, train_data, train_target, test_data, test_target, alpha=0.0024, beta=0.05,  
        iterations=400, tolerance=0.0005, l2_reg=True, verbose=True)

result = model.test()


# # Save model results
# with open('/home/ryu/thesis/new_movielens/state8/FM_model8.pkl', 'wb') as f:
#     pickle.dump(model, f)
