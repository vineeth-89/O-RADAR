#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression
import json
import os

class Plotter:
    def __init__(self, data):
        self.data = data

    def plot_accuracy_vs_sequence_length(self, sequence_lengths, accuracy_scores, title, filename):
        plt.plot(sequence_lengths, accuracy_scores, marker='o')
        plt.title(title)
        plt.xlabel('Sequence Length')
        plt.ylabel('Accuracy')
        plt.grid(True)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="png", bbox_inches="tight", transparent=True)
        plt.clf()

    def plot_vs_packets(self, x, y, title, xlabel, ylabel, filename):
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
        plt.clf()

    def plot_trend_with_spline(self, x, y, title, xlabel, ylabel, filename):
        x_np = np.array(x).reshape(-1, 1)
        y_np = np.array(y)

        regression_model = LinearRegression().fit(x_np, y_np)
        trend_line = regression_model.predict(x_np)

        new_x = np.linspace(min(x), max(x), 300)
        spl = make_interp_spline(x, y, k=3)
        smooth_y = spl(new_x)

        plt.scatter(x, y, label='Original Points', color='green')
        plt.plot(new_x, smooth_y, label='Smoothed Spline')
        plt.plot(x, trend_line, label='Trend Line', color='green', linestyle='--')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format="png", bbox_inches="tight", transparent=True)
        plt.clf()

def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    data = load_data("dataset/plot_data.json")
    plotter = Plotter(data)
    # check if output directory exists, if not create it
    os.makedirs("outputs/figures", exist_ok=True)

    plotter.plot_accuracy_vs_sequence_length(data['sequence_lengths_nas'], data['accuracy_scores_nas'], 
                                             'Accuracy vs Sequence Length (NAS)', 
                                             "outputs/figures/acc-vs-seq-len.png")
    plotter.plot_accuracy_vs_sequence_length(data['sequence_lengths_rrc'], data['accuracy_scores_rrc'], 
                                             'Accuracy vs Sequence Length (RRC)', 
                                             "outputs/figures/acc-vs-seq-len-rrc.png")
    plotter.plot_trend_with_spline(data['number_of_packets'], data['elapsed_time'], 
                                   'Time Consumption', 'Number of Packets', 'Detection Time (ms)', 
                                   "outputs/figures/time-consumption.png")
    plotter.plot_trend_with_spline(data['number_of_packets'], data['memory_usage'], 
                                   'Memory Consumption', 'Number of Packets', 'Memory Usage (KB)', 
                                   "outputs/figures/memory-consumption.png")
    plotter.plot_trend_with_spline(data['number_of_packets'], data['power_consumption'], 
                                   'Power Consumption', 'Number of Packets', 'Power Consumption (mW)', 
                                   "outputs/figures/power-consumption.png")