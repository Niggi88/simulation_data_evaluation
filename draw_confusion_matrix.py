import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from config import load_config
CONFIG = load_config()


"""_summary_
                empty	loaded
target  empty   4000    20	
        loaded	15      3000
                predicted   
                
                empty	loaded
target  empty   TP      FN	
        loaded	FP      TN
                predicted   

"""

def get_accuracy(df: pd.DataFrame):
    tp = pd.to_numeric(df.iloc[1,1])
    tn = pd.to_numeric(df.iloc[0,0])
    total = df.sum().sum()
    return (tp+tn) / total


def calculate_metrics_from_confusion_matrix(df: pd.DataFrame):
    # Assuming df is a confusion matrix with the following format:
    #           Predicted No     Predicted Yes
    # Actual No      TN               FP
    # Actual Yes     FN               TP
    
    TP = pd.to_numeric(df.iloc[0, 0])  # True Positives
    FN = pd.to_numeric(df.iloc[0, 1])  # False Negatives
    FP = pd.to_numeric(df.iloc[1, 0])  # False Positives
    TN = pd.to_numeric(df.iloc[1, 1])  # True Negatives

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def make_confusion_matrix(cm_csv_src: str, title: str) -> None:
    """Creates an confusion matrix image from a csv file containing

    Args:
        cm_csv_src (string): csv file location of confusion matrix image
        cm_png_target (string): target folter where to save confusion matrix
    """
    plt.clf()
    df_cm = pd.read_csv(cm_csv_src, index_col=[0])
    with sns.plotting_context(font_scale=2):
        plt.figure(figsize=(12, 10))
        # sns.set(font_scale=2)
        ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 22}, fmt='d', cmap = "Blues", cbar = False)
        class_labels = [label.get_text() for label in ax.get_yticklabels()]
        ax.set_yticklabels(class_labels, rotation=90)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Target", rotation=90)
        acc = get_accuracy(df_cm)
        precision, recall, f1_score = calculate_metrics_from_confusion_matrix(df_cm)
        title = title + f' T: {CONFIG.confidence_threshold} Precision: {round(precision*100, 2)}%; Recall: {round(recall*100, 2)}%'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(cm_csv_src.with_suffix('.png'), dpi=400)
        plt.close()
    
    
def make_multiclass_confusion_matrix_weighted(cm_csv_src: str, title: str) -> None:
    """Creates an confusion matrix image from a csv file containing

    Args:
        cm_csv_src (string): csv file location of confusion matrix image
        cm_png_target (string): target folter where to save confusion matrix
    """
    
    df_cm = pd.read_csv(cm_csv_src, index_col=[0])
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=2)
    ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 22}, fmt='.2f', cmap = "Blues", cbar = False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    acc = np.trace(df_cm.values) / len(df_cm)
    precision, recall, f1_score = calculate_metrics_from_confusion_matrix(df_cm)
    title = title + f' Precision: {round(precision*100, 2)}%; Recall: {round(recall*100, 2)}%;  f1: {round(f1_score,2)}'
    ax.set_title(title)
    plt.savefig(cm_csv_src.with_suffix('.png'), dpi=400)
    
    
def make_multiclass_confusion_matrix(cm_csv_src: str, title: str) -> None:
    """Creates an confusion matrix image from a csv file containing

    Args:
        cm_csv_src (string): csv file location of confusion matrix image
        cm_png_target (string): target folter where to save confusion matrix
    """
    
    df_cm = pd.read_csv(cm_csv_src, index_col=[0])
    plt.figure(figsize=(15, 7))
    sns.set(font_scale=2)
    ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 22}, fmt='d', cmap = "Blues", cbar = False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    acc = np.trace(df_cm.values) / df_cm.sum().sum()
    precision, recall, f1_score = calculate_metrics_from_confusion_matrix(df_cm)
    title = title + f' Precision: {round(precision*100, 2)}%; Recall: {round(recall*100, 2)}%;  f1: {round(f1_score,2)}'
    ax.set_title(title)
    plt.savefig(cm_csv_src.with_suffix('.png'), dpi=400)


def generate_confusion_matrix(tp, fp, tn, fn, save=True, name="confusion_matrix"):
    true_class, false_class = CONFIG.class_name, "no_" + CONFIG.class_name
    data = {
        '': [true_class, false_class],
        true_class: [tp, fp],
        false_class: [fn, tn]
    }
    df = pd.DataFrame(data)
    df.set_index('', inplace=True)
    confusion_matrix_dir = CONFIG.eval_dir / f"{name}.csv" 
    df.to_csv(confusion_matrix_dir)
    make_confusion_matrix(confusion_matrix_dir, CONFIG.class_name)
    return df

if __name__ == "__main__":
    TITLES = [
        "fresh_food_counter",
        "cigarettes",
        "empty / loaded",
        "bundle / no bundle",
        "n bundles",
    ]

    CONFIG = load_config()
    title = "fresh_food_counter"
    confusion_matrix_source_csv = "/home/niklas/status/models/frischetheke_zigaretten/objectdetector_nano_416x416_frisch_kippen/evaluation/fresh_food_counter/confusion_matrix_session.csv"


    assert title in TITLES, f"Title {title} not in {TITLES}"

    make_confusion_matrix(confusion_matrix_source_csv, title)
    