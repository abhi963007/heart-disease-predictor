"""
Instructions

- Fill the methods and functions that currently raise NotImplementedError.
- The data should be split in test (20%) and training (80%) sets outside the Model
- The Model should predict the target column from all the remaining variables.
- For the modeling methods `train`, `predict` and `eval` you can use any appropriate method.
- Use an appropriate metric based on the data and try to get the best results
- Your solution will be judged on both correctness and code quality.
- Do not modify the part of the code which is below this message -> # You should not have to modify the code below this point
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

class Model:
    def __init__(self, train_df: pd.DataFrame = None, test_df: pd.DataFrame = None) -> None:
        """
        Initialize the Model as necessary

        Args:
            train_df (pd.DataFrame): training data
            test_df (pd.DataFrame): test data
        """
        print("#### Initializing Model ####")
        self.train_df = train_df
        self.test_df = test_df
        self.model = None
        print("Model initialized.\n")

    
    def process_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare the data as needed. If df is None,
        then process the training data passed in the constructor
        and also the test data.

        Args:
            df (pd.DataFrame): data
        """
        # Note: No extensive data preprocessing (e.g., handling null values, scaling numeric columns,
        # or encoding categorical variables) is done here because XGBoost can handle null values, 
        # does not require feature scaling, and can work with categorical features directly. 
        # Performing these preprocessing steps might actually hurt performance.
        
        print("#### Processing Data ####")
        if df is None:
            # Process both train and test data
            self.train_df['sex'] = (self.train_df['sex'] == 'Male').astype(int)
            self.test_df['sex'] = (self.test_df['sex'] == 'Male').astype(int)
            print("Training and test data processed.\n")
        else:
            processed_df = df.copy()
            processed_df['sex'] = (processed_df['sex'] == 'Male').astype(int)
            print("Provided data processed.\n")
            return processed_df

    def train(self) -> None:
        """
        Train a Machine Learning model on the training set passed in the constructor
        """
        print("#### Starting Model Training ####")
        
        X_train = self.train_df.drop('target', axis=1)
        y_train = self.train_df['target']
        X_test = self.test_df.drop('target', axis=1)
        y_test = self.test_df['target']
        
        print("Data split into training and testing sets.")

        # Calculate scale_pos_weight to handle class imbalance
        num_negative = len(y_train[y_train == 0])
        num_positive = len(y_train[y_train == 1])
        scale_pos_weight = num_negative / num_positive

        # Default parameters with some basic tuning
        step_params = {
            'random_state': 42,
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        self.model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                       **step_params, 
                                       early_stopping_rounds=50,
                                       eval_metric=['auc', 'aucpr'])
        self.model.fit(X_train, y_train,
                       eval_set=[(X_train, y_train), 
                                 (X_test, y_test)
                                ], 
                       verbose=100
                      ) 
        
        print("Training completed. Model is ready.\n")

        
    def predict(self, df: pd.DataFrame = None, output_path: str = "./heart_dataset_predictions.csv") -> pd.DataFrame:
        """
        Predict outcomes with the model on the data passed as argument.
        Assumed the data has been processed by the function process_data
        If the argument is None, work on the test data passed in the constructor.
        
        Args:
            df (pd.DataFrame): data
        """
        # If no dataframe is provided, use the test set
        if df is None:
            df = self.test_df
            X = df.drop('target', axis=1)
        else:
            # For prediction data, we don't expect a target column
            X = df

        # Make predictions using the trained model
        predictions = self.model.predict(X)
        prediction_df = pd.DataFrame(predictions, columns=['prediction'])

        # Append predictions to the original DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['prediction'] = prediction_df['prediction']

        if output_path:
            df_with_predictions.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

        return prediction_df
    
    def eval(self, df: pd.DataFrame = None) -> None:
        """
        Evaluate the model in the data passed as argument and print proper metrics.
        And create a short summary of the model you have trained.
        If df is None, then eval the test data passed in the constructor
        
        Args:
            df (pd.DataFrame): data
        """
        print("#### Starting Model Evaluation ####")

         # Determine which dataset to use
        if df is None:
            X_eval = self.test_df.drop('target', axis=1)
            y_eval = self.test_df['target']
        else:
            X_eval = df.drop('target', axis=1)
            y_eval = df['target']

        # Predict outcomes
        predictions = self.model.predict(X_eval)
        
        # Calculate metrics
        accuracy = self.model.score(X_eval, y_eval)
        cm = metrics.confusion_matrix(y_eval, predictions)
        precision = metrics.precision_score(y_eval, predictions)
        recall = metrics.recall_score(y_eval, predictions)
        f1 = metrics.f1_score(y_eval, predictions)
        auc = metrics.roc_auc_score(y_eval, predictions)

        # Print metrics in a table-like format
        print("#### Evaluation Metrics ####")
        print(f"{'Metric':<12} {'Value':<10}")
        print(f"{'-'*12} {'-'*10}")
        print(f"{'Accuracy':<12} {accuracy:.4f}")
        print(f"{'Precision':<12} {precision:.4f}")
        print(f"{'Recall':<12} {recall:.4f}")
        print(f"{'F1 Score':<12} {f1:.4f}")
        print(f"{'AUC':<12} {auc:.4f}")
        print()
        
        # Print the confusion matrix in a readable format
        print("Confusion Matrix:")
        print(f"{'':<12} {'Predicted: 0':<15} {'Predicted: 1':<15}")
        print(f"{'Actual: 0':<12} {cm[0][0]:<15} {cm[0][1]:<15}")
        print(f"{'Actual: 1':<12} {cm[1][0]:<15} {cm[1][1]:<15}")
        print()
        
        # Create directory for plots
        print("#### Saving Evaluation Plots ####")
        plot_dir = 'eval_plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 4))
        cm = metrics.confusion_matrix(y_eval, predictions,
                                      normalize='true')
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        disp.plot(ax=ax, cmap='Blues')
        plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
        plt.close()

        # Save F1 Score Report plot
        fig, ax = plt.subplots(figsize=(8, 4))
        classifier.classification_report(self.model, X_eval, y_eval, ax=ax)
        plt.savefig(os.path.join(plot_dir, 'classification_report.png'))
        plt.close()

        # Save ROC curve plot
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics.RocCurveDisplay.from_estimator(self.model, X_eval, y_eval, ax=ax, label='Test ROC Curve')
        metrics.RocCurveDisplay.from_estimator(self.model, self.train_df.drop('target', axis=1), 
                                            self.train_df['target'], 
                                            ax=ax, label='Train ROC Curve')
        ax.set(title='ROC plots for the model (Train vs. Test)')
        plt.savefig(os.path.join(plot_dir, 'roc_curve_train_test.png'))

        # Feature Importance Plot
        fig, ax = plt.subplots(figsize=(8, 4)) 
        (pd.Series(self.model.feature_importances_, index=X_eval.columns)
        .sort_values()
        .plot.barh(ax=ax))
        plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
        plt.close()
        
        print(f"Evaluation plots are saved in the '{plot_dir}' directory.")
        print("Model evaluation completed.")
    
    def save(self, path: str) -> None:
        """
        Save the model so it can be reused

        Args:
            path (str): path to save the model
        """
        print(f"#### Saving Model to {path} ####")
        self.model.save_model(path)
        print(f"Model saved to {path}\n")

    @staticmethod
    def load(path: str) -> Model:
        """
        Reload the Model from the saved path so it can be re-used.
        
        Args:
            path (str): path where the model was saved to
        """
        print(f"#### Loading Model from {path} ####")

        loaded_model = Model(train_df=None, test_df=None)

        loaded_model.model = xgb.XGBClassifier()
        loaded_model.model.load_model(path)

        print(f"Model loaded from {path}\n")

        return loaded_model



def main():
    
    data_path = "./heart_dataset.csv"
    path_to_save = "./xgb_trained.ubj"
    predict_path = "./heart_dataset_inference.csv"
    output_path = "" 
    
    # Read data 
    df = pd.read_csv(data_path)

    # Split data into training and test set by using df
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    #################################################################################
    # You should not have to modify the code below this point
    
    # Define the model
    model = Model(train_df, test_df)
    
    # Process data
    model.process_data()
    
    # Train model
    model.train()
    
    # Evaluate performance
    model.eval()
    
    # Save model
    model.save(path_to_save)
    
    # Load model
    loaded_model = Model.load(path_to_save)
    
    # Predict results of the predict data
    predict_df = pd.read_csv(predict_path)
    
    predict_df = loaded_model.process_data(predict_df)
    outcomes = loaded_model.predict(predict_df)
    print(f"Predicted on predict data: {outcomes}\n")

if __name__ == '__main__':
    main()
    

