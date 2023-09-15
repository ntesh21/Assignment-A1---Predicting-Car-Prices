import os,sys

# Add parent dir and sub dirs to the python path for importing the modules from different directories
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.extend([rootdir, parentdir])


import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



#experiment tracking
# import mlflow
import os

# # This the dockerized method.
# # We build two docker containers, one for python/jupyter and another for mlflow.
# # The url `mlflow` is resolved into another container within the same composer.
# mlflow.set_tracking_uri("http://localhost:5000")
# # In the dockerized way, the user who runs this code will be `root`.
# # The MLflow will also log the run user_id as `root`.
# # To change that, we need to set this environ["LOGNAME"] to your name.
# os.environ["LOGNAME"] = "nitesh"
# mlflow.create_experiment(name="model_comparision")  #create if you haven't create
# mlflow.set_experiment(experiment_name="nitesh-car-model-comparision")

def compute_r2(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ssr / sst)
    return r2

from sklearn.model_selection import KFold
import math
import random

import warnings
warnings.filterwarnings("ignore")



class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization, lr=0.001, method='batch', weight_init='zero', momentum = False, num_epochs=500, batch_size=50, cv=kfold):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.weight_init = weight_init
        self.momentum = momentum
        self.regularization = regularization
        

    def mse(self, ytrue, ypred):
        return np.mean(((ypred - ytrue) ** 2))#.sum() / ytrue.shape[0]
    
    def fit(self, X_train, y_train):
            
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty

        print("Training shape is",X_train.shape)

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            self.theta = np.zeros(X_cross_train.shape[1])
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, 'weight_init':self.weight_init, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx]
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #record dataset
                    mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
                    mlflow.log_input(mlflow_train_data, context="training")
                    
                    mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
                    mlflow.log_input(mlflow_val_data, context="validation")
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
            
                    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]
        if self.weight_init == 'xavier':
            m = self.initialize(m)  
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        if self.momentum is True:
            self.theta = self.apply_momentum(grad)
        else:
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
    
    def initialize(self, m):
        #This is the added method to initialized the weight with Xavier method
        lower, upper = -(1.0 / math.sqrt(m)), (1.0 / math.sqrt(m))
        numbers = random.randint(1, 1000)
        scaled = lower + numbers * (upper  - lower)
        return scaled


    def apply_momentum(self, gradient, momentum=0.9):
        # This is the added method apply the momentum while training and conversing the gradient
        velocity = np.zeros(X_train.shape[1])
        velocity = momentum * velocity + self.lr  * gradient  
        self.theta -= velocity
        return self.theta


    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    
    def plot_feature_importance(self, features_names):
        # This is the added method to plot the models feature improtance according to coefficient
        if features_names is None:
            raise ValueError("Fit the model before plotting feature importance.")
        # Get the coefficients of the features
        coefficients = self.theta
        scaler = StandardScaler()
        scaler.fit(X_train)
        coefficients = scaler.transform([coefficients])
        # Create a bar plot to visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(features_names, coefficients[0])
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title('Feature Importance based on Coefficients')
        plt.show()
    
    def _bias(self):
        return self.theta[0]


class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        print("This is theta", theta)
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method, weight_init, momentum, lr, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, weight_init, momentum)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, weight_init, momentum,  lr, l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, weight_init, momentum)
        
class ElasticNet(LinearRegression):
    
    def __init__(self, method, weight_init, momentum, lr, l, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, weight_init, momentum)



mileage_median = 19.33
engine_median = 1248.0
max_power_median = 82.4


def calculate_car_age(year):
  #This function calculate the year value by subtraction it with current year and gives total age of the car
  year_built = datetime.date(year, 1, 1)
  year_now = datetime.date.today()
  age = (year_now - year_built).days//365
  return age 

def predict_price_old(year, mileage, engine, max_power=None):
    # load the model from disk
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))
    price_model = pickle.load(open("model/car-price.model", "rb"))#'/apps/model/car-price.model', 'rb'))
    car_age = calculate_car_age(year)
    #Imputation with the median value of training data
    if mileage is None:
       mileage = mileage_median 
    if engine is None:
       engine = engine_median
    if max_power is None:
       max_power = max_power_median
    car_features = np.array([car_age, float(mileage), float(engine), float(max_power)])
    print(car_features)
    # mlflow.start_run(run_name=f"Predict car features old {car_features}", nested=True)
    prediction = price_model.predict([car_features])
    # mlflow.log_metric(key="prediction", value=prediction)
    # signature = mlflow.models.infer_signature([car_features], price_model.predict([car_features]))
    # mlflow.sklearn.log_model(price_model, artifact_path='model-old', signature=signature)
    # mlflow.end_run()
    return prediction

def predict_price_new(year, mileage=None, engine = None, max_power=None):
    # load the model from disk
    print(os.getcwd())
    price_model = pickle.load(open('model/car-price-new.model', 'rb'))
    car_age = calculate_car_age(year)
    #Imputation with the median value of training data
    if mileage is None:
       mileage = mileage_median 
    if engine is None:
       engine = engine_median
    if max_power is None:
       max_power = max_power_median
    car_features = np.array([car_age, float(mileage), float(engine), float(max_power)])
    # mlflow.start_run(run_name=f"Predict car features old {car_features}", nested=True)
    prediction = price_model.predict([car_features])
    # mlflow.log_metric(key="prediction-new", value=prediction)
    # signature = mlflow.models.infer_signature([car_features], price_model.predict([car_features]))
    # mlflow.sklearn.log_model(price_model, artifact_path='model-new', signature=signature)
    # mlflow.end_run()
    return prediction


# print(predict_price_new(2004, 35, 800, 67))