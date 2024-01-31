import os
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, ParameterGrid

import warnings
warnings.filterwarnings(action = 'ignore')

import pickle

import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 불러오기
ratings = pd.read_csv('/home/ryu/thesis/data/amazon/Amazon_ratings.csv')

cnt = ratings.groupby('user_id').count()['rating']
keys = cnt[cnt>3].keys()
ratings = ratings[ratings['user_id'].isin(keys)]

# Rating 데이터를 test, train으로 나누기
x = ratings.copy()
y = ratings['user_id']
ratings_train, ratings_test = train_test_split(x, test_size=0.25, stratify=y, random_state=42)


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
        
#### HYPERPARAMTER TUNING ####

class HyperparameterTuning:
    def __init__(self, model, ratings, param_grid, cv=5):
        self.model = model
        self.ratings = ratings
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = float('inf')
    
    def fit(self, ratings, y=None):
        for params in ParameterGrid(self.param_grid):
            print("Parameters:", params)
            rmse_scores = []

            for i in range(self.cv):
                x = ratings.copy()
                y = ratings['user_id']
                ratings_train, ratings_test = train_test_split(x, test_size=0.25, stratify=y)

                R_temp = ratings_train.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
                
                model = self.model(R_temp, **params)
                model.set_test(ratings_test)
                _, best_rmse = model.train()
                
                rmse_scores.append(best_rmse)
            
            avg_rmse = np.mean(rmse_scores)
            
            print("*****************************")
            print(f"Average RMSE for parameters {params}: {avg_rmse:.6f}\n")
            print("*****************************")

            if avg_rmse < self.best_score_:
                self.best_score_ = avg_rmse
                self.best_params_ = params

        print("Finished evaluating all parameter combinations.")
        print("\nBest Parameters:")
        for param, value in self.best_params_.items():
            print(f"{param}: {value}")
        print(f"Best RMSE: {self.best_score_:.6f}")


# mf = NEW_MF(R_temp, K=250, alpha=0.0007, beta=0.003, iterations=600, tolerance=0.0001, verbose=True)
param_grid = {
    'K': [220],
    'alpha': [0.0014],
    'beta': [0.001, 0.002 ,0.003, 0.004, 0.005],
    'iterations': [200],
    'verbose': [True]
}

mf_ht = HyperparameterTuning(MF_base, ratings, param_grid, cv=3)
mf_ht.fit(ratings)