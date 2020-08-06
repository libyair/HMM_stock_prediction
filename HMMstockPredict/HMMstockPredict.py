import os
import click
import tkinter.messagebox
import pandas as pd
from data_analysis import StockPredictor
import pickle
import numpy as np
import traceback
import datetime
import csv

@click.command()
@click.option('--path', '-p', help="The path to the file and file name containing the data")
@click.option('--stock-name', '-s', type=(str), multiple=True,
    help="The name of the stock to be predicted. Except stock-name=all to for entire file prediction",
    default='all')
@click.option('--mode', '-m', type=click.Choice(['predict', 'train'], case_sensitive=False), help="choose mode='train' to train and save a new model, or mode='predict' for predict for current day")
@click.option('--with-plot', '-t', default= True, help="Control on plotting in the end of each run")
@click.option('--set-dayindex', '-d',  default=None , help="Control day index")
def run(path,stock_name, mode, with_plot,set_dayindex):
    try:
        df = pd.read_csv(f'{path}', index_col=None, header=0)
        
        #Check Data:
        assert len(df)>=10, "Not enguph data for prediction, at least 10 consecutives records needed"

        print(f"Upload finished succesfully")
        
        folder_path = os.path.dirname(os.path.dirname(path)) #get the project directory
        # Create results directory
        if not os.path.exists(folder_path+'/results'):
            os.makedirs(folder_path+'/results')

        if not os.path.exists(folder_path+'/trained_models'):
            os.makedirs(folder_path+'/trained_models')
        
        
        if 'all' in stock_name:
            open_col = [col for col in df.columns if '-open' in col]
            stock_name = [col.replace('-open', "") for col in open_col]
               
        
        print("Got the following stock name: "+str(stock_name))
        
        prediction_dict = {}
        MAPE_score = {}
        if mode == 'train':
            print("training mode")
            for stock in stock_name: 
                stock_cols = [col for col in df.columns if stock in col]
                stock_data = df[stock_cols]
                stock_data.columns = stock_data.columns.str.replace(stock+'-', '')   
                stock_data['Date'] = df['Date']
                

                stock_predictor = StockPredictor(stock_data, stock, folder_path, mode)
                           
                stock_predictor.fit()          
                # save model
                with open(f"{folder_path}/trained_models/{stock}_model.pkl", "wb") as file: pickle.dump(stock_predictor.hmm, file)
                print(f"A model for {stock} was saved in folder {folder_path}\trained_models")
                
                if set_dayindex is None:
                    days_index = len(stock_predictor._test_data)
                else: 
                    days_index = int(set_dayindex)
                
                stock_predictor.predict_close_prices_for_days(days_index, with_plot=with_plot)
                MAPE_score[stock] = stock_predictor.mape_score

            
            today = datetime.datetime.strftime(datetime.date.today(), '%m/%d/%Y')
            with open(f"{folder_path}/results/training_MAPE_score.csv", 'w') as f:
                f.write("%s,%s\n\n"%('Running date',today))
                f.write("%s,%s\n"%('Stock name','MAPE score'))
                for key in MAPE_score.keys():
                    f.write("%s,%f\n"%(key,MAPE_score[key]))

        elif mode == 'predict':
            print("Prediction mode")
            for stock in stock_name: 
                stock_cols = [col for col in df.columns if stock in col]
                stock_data = df[stock_cols]
                stock_data.columns = stock_data.columns.str.replace(stock+'-', '')   
                stock_data['Date'] = df['Date']
                

                stock_predictor = StockPredictor(stock_data, stock, folder_path, mode)
                        
                stock_predictor.predict_close_prices_for_current(with_plot=with_plot)
                
                prediction_dict[stock] = stock_predictor.predicted_close
            
                       
            with open(f"{folder_path}/results/{stock_predictor.predict_date_str}_prediction.csv", 'w') as f:
                f.write("%s,%s\n\n"%('Date',stock_predictor.predict_date_str))
                f.write("%s,%s\n"%('Stock name','Close Value'))
                for key in prediction_dict.keys():
                    f.write("%s,%f\n"%(key,prediction_dict[key]))
            
            print("Prediction Complete")

    except Exception as e:
        # hide main window
        root = tkinter.Tk()
        root.withdraw()
        tkinter.messagebox.showerror(title="Error message", message=e)
        print(e)
        traceback.print_exc()
                

if __name__ == "__main__":
    run()