This project is predicting stock prices in one following day using HMM model based on open, close, high and low values in the past 15 years.

To run the project, please download the folder content into your PC. 

*Requirments*:
1. To Run the predictor, a PC/MAC must have the following installed:
    1.1. Python 3.7 or above installed with conda and pip.
    1.2. Git instalation with Github account. Download from the following link: https://git-scm.com/download/win.
    1.3. Microsoft Visual C++ 14.0. Download from the following link: https://visualstudio.microsoft.com/downloads/.
2. Tamplates for data for traing/prediction can be found in the *data* folder in the project directory. These are the only stractured that  are supported. 

*Instruction for creating predictions using command line:*
1. In command line terminal go to the code folder in the project directory. 
2. Install requirementas by running the line: pip install -r HMMstockPredict/requirements.txt
3. Install hmmlearn library by by running the line: pip install --user git+https://github.com/hmmlearn/hmmlearn
3. The following command will run prediction with a suitable data file: 
        python HMMstockPredict/HMMstockPredict.py --stock-name all --path "path_to_file/file.csv" --mode predict --with-plot False
4. After running, the service will print the predicted results for each stock listed in in stock-name, and save it in the results folder in the path. 

5. Command options:
    - --mode, -m : choose mode=train to train and save a new model, or mode=predict for predict for current day.
    - --path, -p, The path to the data file, including full file name.
    - --stock-name, -s, The name of the stock to be predicted. Excepts stock-name=all to for entire file prediction.
    - --with-plot, -t, default= True, Controls on plotting in the end of each run.

*Instruction for retraining the model using command line:*

1. To retrin the model using data in the same format as the data file, run the following command:
    python code/HMMstockPredict.py --stock-name all --path "path_to_data_file*.csv" ' --mode "train" --with-plot True
2. After fitting the model, the service will run prediction on a testing set, and will calculate MAPE score for the data. 
3. At the end of the run the model will be saved in the model directory under the name "<stock_name>_model.pkl"
4. Final MAPE result for all runs will be save in a csv file in the 'result' folder. 


*Results in MAPE score*:

S: 0.845293787676171
T: 0.6744336266434269
G: 0.5640196809404087
SHY: 0.20477107219535998
IEF: 0.2871859321018051 
LQD: 0.46555319226156644
VIX: 5.60083