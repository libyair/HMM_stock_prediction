"""
Usage: analyse_data.py --company=<company>
"""
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from hmmlearn.hmm import GMMHMM
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
import holidays
import pickle


# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
# Change plot style to ggplot (for better and more aesthetic visualisation)
plt.style.use('ggplot')
 
 
class StockPredictor(object):
    def __init__(self, data, stock_name, folder_path, mode, test_size=0.2,
                 n_hidden_states=4, n_mix_components= 5, n_latency_days=10,
                 n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10):
 
        self._init_logger()
        self.stock_name = stock_name
        self.n_latency_days = n_latency_days
        self.n_hidden_states = n_hidden_states
        self.n_mix_components = n_mix_components
        self.folder_path = folder_path
        if mode == 'train': 
            #self.hmm = GMMHMM(n_components=n_hidden_states, n_mix=n_mix_components, init_params="mcw")
            self.hmm = GaussianHMM(n_components=n_hidden_states, init_params="mcw")
            self._split_train_test_data(data, test_size=0.05)
        else:
            with open(f"{folder_path}/trained_models/{stock_name}_model.pkl", "rb") as file: self.hmm = pickle.load(file)
            
            if np.isnan(data['close'].iloc[0]): 
                self.target_day_open = data['open'].iloc[len(data)-1]
                self.prediction_date = data['Date'].iloc[len(data)-1]
                self._test_data = data.drop(len(data)-1)
                
            else:
                self.target_day_open = data['close'].iloc[len(data)-1]
                self.prediction_date = self._next_busyness_day(data['Date'].iloc[len(data)-1])
                self._test_data = data
            
            self.predict_date_str = datetime.datetime.strftime(self.prediction_date, '%m-%d-%Y')  

        self._compute_all_possible_outcomes(
            n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)
        
         
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
 
    def _split_train_test_data(self, data, test_size):
        _train_data, test_data = train_test_split(
            data, test_size=test_size, shuffle=False)
        self._train_data = _train_data
        self._test_data = test_data
 
    @staticmethod
    def _extract_features(data):
        open_price = np.array(data['open'])
        close_price = np.array(data['close'])
        high_price = np.array(data['high'])
        low_price = np.array(data['low'])
 
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price
        return np.column_stack((frac_change, frac_high, frac_low))
 
    def _next_busyness_day(self, day):
        ONE_DAY = datetime.timedelta(days=1)
        HOLIDAYS_US = holidays.US()
        next_day = datetime.datetime.strptime(day, '%m/%d/%Y') + ONE_DAY
        while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
            next_day += ONE_DAY
        return next_day

    def fit(self):
        self._logger.info('>>> Extracting Features')
        feature_vector = StockPredictor._extract_features(self._train_data)
        self._logger.info('Features extraction Completed <<<')

        self.hmm.fit(feature_vector)
        self._logger.info(self.hmm.monitor_)
        self._logger.info(self.hmm.monitor_.converged)
 
    def _compute_all_possible_outcomes(self, n_steps_frac_change,
                                       n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)
 
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))
 
    def _get_most_probable_outcome(self, day_index):
        self.previous_data_start_index = max(0, day_index - self.n_latency_days)
        self.previous_data_end_index = max(0, day_index - 1)
        
        previous_data = self._test_data.iloc[self.previous_data_end_index: self.previous_data_start_index]
        previous_data_features = StockPredictor._extract_features(
            previous_data)
 
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]
 
        return most_probable_outcome
 
    def predict_close_price(self, day_index, open_price):
        predicted_frac_change, _, _ = self._get_most_probable_outcome(
            day_index)
        return open_price * (1 + predicted_frac_change)
 
    def mape_estimate(self, y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 
    
     
    
    def predict_close_prices_for_days(self, days, with_plot=False):
        predicted_close_prices = []
        for day_index in tqdm(range(days)):
            open_price = self._test_data.iloc[day_index]['open']
            predicted_close_prices.append(self.predict_close_price(day_index, open_price))
        #compute score
        test_data = self._test_data[0: days]
        actual_close_prices = test_data['close']
        self.mape_score  = self.mape_estimate(actual_close_prices, predicted_close_prices)
        print(f"MAPE score for {self.stock_name}="+str(self.mape_score))
        # plot results
        
        days = np.array(test_data['Date'])# dtype="datetime64[ms]")
        actual_close_prices = test_data['close']
        if with_plot:
            fig = plt.figure()

            axes = fig.add_subplot(111)
            axes.plot(days, actual_close_prices, 'bo-', label="actual")
            axes.plot(days, predicted_close_prices, 'r+-', label="predicted")
            axes.set_title('{stock_name}'.format(stock_name=self.stock_name))

            fig.autofmt_xdate()

            plt.legend()
            plt.show(block=False)
          
            plt.savefig(f"{self.folder_path}/results/{self.stock_name}_test_prediction.png")
        

        return predicted_close_prices

    def predict_close_prices_for_current(self, with_plot=False ):
        self.predicted_close = self.predict_close_price(len(self._test_data), self.target_day_open )
        print(f"""{self.stock_name} predicted price for {datetime.datetime.strftime(self.prediction_date, '%m/%d/%Y')} is {self.predicted_close}""")            
        close_prices = self._test_data['close']
        if with_plot:
            dates_plt = dates.date2num(pd.to_datetime(self._test_data['Date'], format='%m/%d/%Y'))
                
            fig = plt.figure()

            axes = fig.add_subplot(111)
            axes.plot(dates_plt[self.previous_data_start_index:self.previous_data_end_index], 
                close_prices[self.previous_data_start_index:self.previous_data_end_index], 'bo-', label="actual")
            axes.plot(dates.date2num(self.prediction_date), self.predicted_close, 'r+-', label="predicted")
            axes.set_title('{stock_name}'.format(stock_name=self.stock_name))
            fig.autofmt_xdate()

            plt.legend()
            #plt.show(block=False)
            plt.savefig(f"{self.folder_path}/results/{self.stock_name}_prediction_{self.predict_date_str}.png")
            



 
