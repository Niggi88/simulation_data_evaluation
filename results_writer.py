from config import load_config
from datetime import datetime
import numpy as np
from pathlib import Path


class ResultsWriter:
    def __init__(self):
        config = load_config()
        self.dir = Path(config.out_dir) / "results.txt"
        self.results = {}
        with open(self.dir, "w") as f:
            f.write(f"Results - {datetime.now()}\n")
        
    def add_results(self, class_name, tp, fp, tn, fn):
        with open(self.dir, "a") as f:
            f.write(f"{class_name} - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            f.write(f"Precision: {precision}, Recall: {recall}, F1: {f1}\n")
            f.write("\n")
            
        if class_name not in self.results:
            self.results[class_name] = {'precisions': [], 'recalls': []}
        self.results[class_name]['precisions'].append(precision)
        self.results[class_name]['recalls'].append(recall)
        
    def compute_map(self):
        average_precisions = []
        for class_name, metrics in self.results.items():
            precisions = np.array(metrics['precisions'])
            recalls = np.array(metrics['recalls'])
            
            sorted_indices = np.argsort(recalls)
            precisions = precisions[sorted_indices]
            recalls = recalls[sorted_indices]
            
            average_precision = np.mean(precisions)
            average_precisions.append(average_precision)
        
        mean_average_precision = np.mean(average_precisions)
        return mean_average_precision
        
    def write_map(self):
        map_value = self.compute_map()
        with open(self.dir, "a") as f:
            f.write(f"Mean Average Precision (MAP): {map_value}\n")