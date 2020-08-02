import requests
import os
import glob
import click
import tkinter.messagebox
import pandas as pd
from data_analysis import StockPredictor
import pickle
import numpy as np
import traceback

@click.command()
@click.option('--path', '-p', help="The path to the file and file name containing the data")
@click.option('--stock-name', '-s', help="The name of the stock ato be predicted. Except stock-name=all to for entire file prediction",
    default='all')
@click.option('--mode', '-m', help="mode='train' to train and save a new model, and mode='predict' for predict for current day")
@click.option('--model-name', '-n', default= None, help="Name of the pre-trained hmm model")
@click.option('--with-plot', '-t', default= True, help="Control on plotting in the end of each run")
# @click.option('--last-close', '-l', default= False, help="Flag detrmine if the closing price for the target day is included in data")


def run(path,stock_name, mode, model_name, with_plot):
    try:
        df = pd.read_csv(f'{path}', index_col=None, header=0)
        
        #Check Data:
        assert len(df)>=10, "Not enguph data for prediction, at least 10 consecutives records needed"


        folder_path = os.path.dirname(path)
        # Create results directory
        if not os.path.exists(folder_path+'/results'):
            os.makedirs(folder_path+'/results')
        
        if stock_name == 'all':
            open_col = [col for col in df.columns if '-open' in col]
            stock_names = [open_col.replace('-open', "") for col in open_col]
            print("Got the following stock names: "+str(stock_names)) ## TODO -remove brackets from print
        else:
            stock_names = [stock_name]
            print("Got the following stock name: "+str(stock_name)) 
        
        for stock in stock_names: 
            stock_cols = [col for col in df.columns if stock in col]
            stock_data = df[stock_cols]
            stock_data.columns = stock_data.columns.str.replace(stock+'-', '')   
            stock_data['Date'] = df['Date']
            print(stock_data.columns)
            print("Upload finished succesfully")

            stock_predictor = StockPredictor(stock_data, stock, folder_path, mode, model_name)
            
            if mode == 'train':
                print("training mode")
                stock_predictor.fit()          
                # save model
                with open(f"{folder_path}\{stock}_model.pkl", "wb") as file: pickle.dump(stock_predictor.hmm, file)
                print("Model was saved in folder "+ folder_path)
                days_index = len(stock_predictor._test_data)
                stock_predictor.predict_close_prices_for_days(days_index, with_plot)
            else:
                # if there is no open price for target day - clone the closing day or the previous day
                print("Prediction mode")
                stock_predictor.predict_close_prices_for_current(with_plot)
                
     

    except Exception as e:
        # hide main window
        root = tkinter.Tk()
        root.withdraw()
        tkinter.messagebox.showerror(title="Error message", message=e)
        print(e)
        traceback.print_exc()
                



if __name__ == "__main__":
    run()