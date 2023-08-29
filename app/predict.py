import pickle
import datetime
import numpy as np
import pandas as pd

mileage_median = 19.33
engine_median = 1248.0
max_power_median = 82.4


def calculate_car_age(year):
  #This function calculate the year value by subtraction it with current year and gives total age of the car
  year_built = datetime.date(year, 1, 1)
  year_now = datetime.date.today()
  age = (year_now - year_built).days//365
  return age 

def predict_price(year, mileage=None, engine = None, max_power=None):
    # load the model from disk
    price_model = pickle.load(open('./model/car-price.model', 'rb'))
    car_age = calculate_car_age(year)
    #Imputation with the median value of training data
    if mileage is None:
       mileage = mileage_median 
    if engine is None:
       engine = engine_median
    if max_power is None:
       max_power = max_power_median
    car_features = np.array([car_age, float(mileage), float(engine), float(max_power)])
    prediction = price_model.predict([car_features])
    return prediction


