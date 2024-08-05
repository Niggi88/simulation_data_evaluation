import os
import numpy as np
import onnxruntime as rt
from pprint import pprint
import pathlib
from tqdm import tqdm
from loguru import logger
import pandas as pd
from dataset.train_set import TrainSet
from dataset.simulation_set import SimulationSet


from data_processing import predict, get_class_map, compute_iou_single, prediction_to_xyxy_box
from draw_confusion_matrix import make_confusion_matrix
from draw.draw_trainset_boxes import draw_detections
import cv2

from results_writer import ResultsWriter
from config import load_config

logger.add(pathlib.Path("logs") / "results.log", rotation="1 MB")


if 'CUDAExecutionProvider' in rt.get_available_providers():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

sess_options = rt.SessionOptions()
sess_options.log_severity_level = 4
sess_options.add_session_config_entry('session.save_model_format', 'ORT')
sess_options.add_session_config_entry('protected_namespaces', '()')

writer = ResultsWriter()

# image kram
def find_prediction(predictions, label):
    class_index = label["class_index"]
    lbl_box = label["box"]
    for idx, prediction in enumerate(predictions):

        if prediction["class_index"] == class_index:
            iou = compute_iou_single(prediction_to_xyxy_box(lbl_box), prediction_to_xyxy_box(prediction["box"]))
            if iou > IOU_THRESHOLD:
                label["prediction"] = prediction
                return idx
    return -1
                

def evaluate_image(image_dir, labels):
    predictions = predict(SESS, image_dir, CLASS_MAP)
    for label in labels:
        prediction_id = find_prediction(predictions, label)
        if prediction_id != -1:
            label["prediction"] = predictions.pop(prediction_id)
    
    return labels, predictions


def metrics_from_labels_and_predictions(labels, predictions, class_name=None):
    fn = 0
    tp = 0
    if class_name is not None:
        labels = [label for label in labels if label["class_name"] == class_name]
        predictions = [prediction for prediction in predictions if prediction["class_name"] == class_name]
    error_class = None
    for label in labels:
        if "prediction" not in label:
            fn += 1 
            error_class = label["class_name"]
        else: 
            tp += 1
    fp = int(len(predictions) > 0)
    tn = int(len(labels) == 0)
    if fp > 0:
        error_class = predictions[0]["class_name"]
    return tp, fp, tn, fn, error_class


def generate_confusion_matrix(tp, fp, tn, fn, save=True, name="confusion_matrix", class_name="correct_box"):
    print("tp", tp)
    true_class, false_class = class_name, "no_" + class_name
    data = {
        '': [true_class, false_class],
        true_class: [tp, fp],
        false_class: [fn, tn]
    }
    df = pd.DataFrame(data)
    df.set_index('', inplace=True)
    confusion_matrix_dir = CONFIG.out_dir / f"{name}.csv"
    df.to_csv(confusion_matrix_dir)
    make_confusion_matrix(confusion_matrix_dir, name)
    return df


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def evaluate(data_set):
    errors = 0
    fps = 0
    fns = 0
    boths = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    results_per_class = {}
    for i, cl in CLASS_MAP.items(): 
        results_per_class[cl] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        
    
    if not os.path.exists(OUT_IMAGES):
        os.makedirs(OUT_IMAGES, exist_ok=True)
    with tqdm(range(len(data_set)   ), desc="Processing") as pbar: # len(data_set)
        for i in pbar:
            img, labels = data_set[i]
            labels, predictions = evaluate_image(img, labels)
            
            for i, cl in CLASS_MAP.items(): 
                tp_, fp_, tn_, fn_, error_class = metrics_from_labels_and_predictions(labels, predictions, cl)
                results_per_class[cl]["tp"] += tp_
                results_per_class[cl]["tn"] += tn_
                results_per_class[cl]["fp"] += fp_
                results_per_class[cl]["fn"] += fn_
            tp_, fp_, tn_, fn_, error_class = metrics_from_labels_and_predictions(labels, predictions)
            prediction_error = max(fp_, fn_) > 0
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_
            if prediction_error:
                assert fp_ > 0 or fn_ > 0
                error_type = "fp" if fp_ > 0 else "fn"
                error_type = "both" if fp_ > 0 and fn_ > 0 else error_type
                assert error_class is not None
                
                out_images = os.path.join(OUT_IMAGES, error_class, error_type, os.path.basename(img.replace(".jpeg", ".png")))
                ensure_dir(os.path.dirname(out_images))
                draw_detections(img, predictions, labels, out_images)
                if error_type == "fp": fps += 1
                if error_type == "fn": fns += 1
                if error_type == "both": boths += 1
                pbar.set_postfix(fp=fps, fn=fns, both=boths, refresh=True)

    generate_confusion_matrix(tp, fp, tn, fn, save=True, name="confusion_matrix")
    writer.add_results("all", tp, fp, tn, fn)
    for i, cl in CLASS_MAP.items(): 
        generate_confusion_matrix(results_per_class[cl]["tp"], results_per_class[cl]["fp"], results_per_class[cl]["tn"], results_per_class[cl]["fn"], save=True, name=f"confusion_matrix_{cl}", class_name=cl)
        writer.add_results(cl, results_per_class[cl]["tp"], results_per_class[cl]["fp"], results_per_class[cl]["tn"], results_per_class[cl]["fn"])
    print(errors / 10) 


if __name__ == "__main__":
    CONFIG = load_config()
    IOU_THRESHOLD = CONFIG.iou_threshold
    data_set = SimulationSet(CONFIG.folder)
    data_set.profile_set()
    assert os.path.isfile(CONFIG.model_dir), f"Model directory {CONFIG.model_dir} does not exist."
    
    OUT_IMAGES = CONFIG.out_dir / "images" # os.path.join(OUT, "images")
    sess_options.enable_profiling = True
    SESS = rt.InferenceSession(CONFIG.model_dir, 
                               session_options=sess_options,
                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) #  'CUDAExecutionProvider'
    CLASS_MAP = get_class_map(CONFIG.model_dir)
    evaluate(data_set)
