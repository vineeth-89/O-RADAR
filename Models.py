import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import importlib
import os
import time

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, log_loss
)

from sklearn.linear_model import(
    LogisticRegression
)

import tensorflow as tf

from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ORAN_Helper import Metric, Processor, Plotter

class MLP(nn.Module):
    def __init__(self, number_of_features, learning_rate, dataloader):
        super(MLP, self).__init__()

        self.dataloader = dataloader
        self.input_dimension = number_of_features

        self.model = nn.Sequential(
            nn.Linear(self.input_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

        self.loss_function = nn.BCELoss()
        # self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using device: {self.device}")
    
    def forward(self,x):
        return self.model(x)

    def fit(self,X_train, y_train, epochs):
        start_time = time.time()
        print_num = epochs // 10

        self.best_accuracy = 0
        self.best_epoch = 0

        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(1, epochs + 1):
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.loss_function(out,y)
            loss.backward()
            self.optimizer.step()

            preds = (out > 0.5).int().cpu().numpy()
            y_true = y.cpu().numpy()
            accuracy = accuracy_score(y_true, preds)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = epoch

            if epoch == 1 or epoch % print_num == 0 or epoch == epochs:
                print(f"Epoch {epoch} ---->>>>>>>>>>, Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.4f}")
                print()

        end_time = time.time()

        self.time_taken = end_time - start_time

        print(f"\nTime Taken    : {self.time_taken:.2f} sec")
        print(f"--Final Accuracy    : {self.time_taken:.2f} sec")
        print(f"--Best Accuracy: {self.best_accuracy * 100}, epoch no. {self.best_epoch}")
        

    def predict_proba(self, X_test):
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.forward(X_test).cpu().numpy()
        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return (probs > 0.5).astype(int)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Defining Metrics for this model
        self.metrics = Metric(accuracy=acc, y_test=y_test, y_pred=y_pred,time_taken=self.time_taken)

        return self.metrics
    
class LR():
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train, epochs=100,validation_size=0.1):
        X_train,X_val,y_train, y_val = train_test_split(X_train, y_train,test_size=validation_size, random_state=42)
        self.model = LogisticRegression(random_state=42, max_iter=100)

        training_loss = []
        validation_loss = []

        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.fit(X=X_train, y=y_train)

            y_train_pred = self.model.predict_proba(X_train)[:,1]
            y_val_pred = self.model.predict_proba(X_val)[:,1]

            train_loss = log_loss(y_train, y_train_pred)
            val_loss = log_loss(y_val, y_val_pred)

            training_loss.append(train_loss)
            validation_loss.append(val_loss)           

        end_time = time.time()

        print(f"Len of ttaining loss: {training_loss}")

        self.time_taken = end_time - start_time

        print(f"Time taken by the model: {self.time_taken}")

        self.plot_curves(training_loss=training_loss, validation_loss=validation_loss)
    
    def plot_curves(self, training_loss, validation_loss):
        plt.plot(list(range(1, len(training_loss) + 1)), training_loss, label="Training Loss", color='blue')
        plt.plot(list(range(1, len(training_loss) + 1)), validation_loss, label="Validation Loss", color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Log-Loss")
        plt.title("Training vs Validation Loss Curve")
        plt.legend()
        plt.show()

    def predict(self, X_test):
        y_pred = self.model.predict(X=X_test)

        return y_pred

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Defining Metrics for this model
        self.metrics = Metric(accuracy=acc, y_test=y_test, y_pred=y_pred,time_taken=self.time_taken)

        return self.metrics
    

class LSTM():
    def __init__(self):
        pass
        
