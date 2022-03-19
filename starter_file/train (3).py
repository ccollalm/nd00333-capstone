from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()

    y_df = x_df.pop("DEATH_EVENT")
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    args = parser.parse_args()

    run = Run.get_context()
    ws = run.experiment.workspace
    found = False
    key = "hyperdrive-heart_failure_data"
    description_text = "Heart Failure Clinical Records Dataset from Kaggle"
    
    if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Datset: ",dataset.take(5).to_pandas_dataframe())
    

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    # ds = ### YOUR CODE HERE ###
    
    
    x, y = clean_data(dataset)

    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###
    #Splittin 80% Training 20% validation
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    auc=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    
    
    #auc = model.score(x_test, y_test)
    run.log("AUC",np.float(auc))
    #accuracy = model.score(x_test, y_test)
    #run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()