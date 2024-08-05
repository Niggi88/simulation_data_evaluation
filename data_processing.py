import onnxruntime as rt
import numpy as np
import cv2
import numpy as np
import onnx
from loguru import logger
from config import load_config

# Create a runtime InferenceSession
print(cv2.__version__)


CONFIG = load_config()



def get_class_map(model_dir):
    model = onnx.load_model(model_dir)
    meta_data = model.metadata_props
    class_map = None
    for item in meta_data:
        if item.key == 'names':
            class_map = item.value
    if class_map is None: raise Exception("No class map found")
    
    # class_map = meta_data["names"]
    return eval(class_map)# {item.key: item.value for item in meta_data}


def apply_class_specific_non_max_suppression(predictions, iou_threshold):
    # If no detections, return an empty list
    if len(predictions) == 0:
        return []

    # Group predictions by class
    boxes_by_class = {}
    for box in predictions:
        class_label = np.argmax(box[5:])  # Assuming class label is at index 5
        if class_label not in boxes_by_class:
            boxes_by_class[class_label] = []
        boxes_by_class[class_label].append(box)

    final_boxes = []

    # Apply NMS for each class
    for class_label, boxes in boxes_by_class.items():
        # Convert bounding boxes to format (x1, y1, x2, y2)
        converted_boxes = np.array([[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes])
        scores = np.array([box[4] for box in boxes])

        selected_boxes = []
        idxs = np.argsort(scores)[::-1]

        while len(idxs) > 0:
            selected_boxes.append(idxs[0])
            ious = compute_iou(converted_boxes[idxs[0]], converted_boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_threshold]

        final_boxes.extend([boxes[idx] for idx in selected_boxes])

    return final_boxes


def prediction_to_xyxy_box(prediction):
    x = prediction["x"]
    y = prediction["y"]
    w = prediction["w"]
    h = prediction["h"]
    return [x-0.5*w, y-0.5*h, x + 0.5*w, y + 0.5*h]


def non_max_suppression(predictions, iou_threshold):
    # If no detections, return an empty list
    if len(predictions) == 0:
        return []

    # Convert bounding boxes to format (x1, y1, x2, y2)
    # as it's a more convenient format for the NMS operation
    boxes = np.array([[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in predictions])
    scores = np.array([box[4] for box in predictions])

    # List to store the indices of the boxes we pick
    selected_boxes = []

    # Get the indices of boxes sorted by confidence score (in decreasing order)
    idxs = np.argsort(scores)[::-1]

    # Loop until the list of box indices is empty
    while len(idxs) > 0:
        # Take the first box and add its index to the list of picked indices
        # Since we sorted the indices, this is the box with the highest confidence score
        selected_boxes.append(idxs[0])

        # Compare IoU (Intersection over Union) of this box with all others
        ious = compute_iou(boxes[idxs[0]], boxes[idxs[1:]])

        # Only keep boxes that are not significantly overlapping with
        # the box we just picked (i.e., IoU < threshold)
        idxs = idxs[1:][ious < iou_threshold]

    return [predictions[idx] for idx in selected_boxes]


def compute_iou_single(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection of two boxes
    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])

    # Compute the area of intersection
    intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # Compute the area of both boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the IoU
    iou = intersection / (box1_area + box2_area - intersection)

    return iou


def compute_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection of two boxes
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])

    # Compute the area of intersection
    intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # Compute the area of both boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    # Compute the IoU
    iou = intersection / (box1_area + box2_area - intersection)

    return iou


def filter_by_confidence(predictions, confidence_threshold):
    # Assuming your confidence score is at index 4
    predictions = predictions[0,0,:,:]

    class_confidence = np.max(predictions[:,5:], axis=1)
    box_confidence = predictions[:,4]
    total_confidence = box_confidence * class_confidence

    #TODO: make sure this is correct
    
    predictions = predictions[total_confidence >= confidence_threshold]
    
    return predictions


def read_image_from_file(image_dir):
    return cv2.imread(image_dir)


def preprocess_image(image, input_shape):
    image = np.array(image, dtype=np.float32)
    # get input shape
    _, _, height, width = input_shape
    
    # Resize the image
    image = cv2.resize(image, (height, width))

    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert from HWC to CHW
    image = image.transpose((2, 0, 1))

    # Convert to float and normalize if needed
    image = image.astype(np.float32) / 255.0

    # Add an extra dimension for batch size
    image = np.expand_dims(image, axis=0)
    
    return image

def is_edge(x,y,w,h):
    return x-0.5*w <= 0.01 or y-0.5*h <= 0.01 or x+0.5*w >= 0.99 or y+0.5*h >= 0.99

def process_results(results, conf_thresh, iou_thresh, class_map, rescale_dims):
    predictions = filter_by_confidence(results, confidence_threshold=conf_thresh)
    
    predictions = apply_class_specific_non_max_suppression(predictions, iou_threshold=iou_thresh)
    
    detections = []
    for pred in predictions:
        box = pred[:4].tolist()
        x, y, w, h = box
        rescale_x = 1 / rescale_dims[0]
        rescale_y = 1 / rescale_dims[1]
        x, w = x * rescale_x, w * rescale_x
        y, h = y * rescale_y, h * rescale_y

        class_index = int(np.argmax(pred[5:]))
        confidence = float(pred[4]) * float(pred[5 + class_index])
        class_name = class_map[class_index]
        detections.append({"box": {'x': x, 'y': y, 'w': w, "h": h}, "confidence": confidence, "class_index": class_index, "class_name": class_name, "is_border": is_edge(x,y,w,h), "confidence": confidence, "box_confidence": float(pred[4]), "class_confidence": float(pred[5 + class_index])})
    
    return detections


############### LETTERBOXING TEST ################
def preprocess_image_letterbox(image, input_shape):
    image = np.array(image, dtype=np.float32)
    height, width, _ = image.shape
    
    
    # Get input shape
    _, _, input_height, input_width = input_shape
    
    # Resize while maintaining aspect ratio (letterboxing)
    scale = min(input_width / width, input_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    original_image_shape = resized_image.shape
    
    # Create a blank canvas
    canvas = np.zeros((input_height, input_width, 3), dtype=np.float32)
    
    # Paste the resized image onto the canvas
    y_offset = (input_height - new_height) // 2
    x_offset = (input_width - new_width) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    # Convert from BGR to RGB
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Convert from HWC to CHW
    canvas = canvas.transpose((2, 0, 1))

    # Convert to float and normalize if needed
    canvas = canvas.astype(np.float32) / 255.0

    # Add an extra dimension for batch size
    canvas = np.expand_dims(canvas, axis=0)
    
    return canvas, original_image_shape

# def process_results          (results, conf_thresh, iou_thresh, class_map, rescale_dims, input_shape, original_image_shape):

def convert_v8_to_v5(input_array):
    """Converts the output of YOLOv8 to the same format as YOLOv5
    in yolov8 the class confidences are combined with the box confidences
    result is (x, y, x, y, bc*pcl1, bc*pcl2, ..., bc*pclN)
    in yolov5 the class confidences
    result is (x, y, x, y, bc, pcl1, pcl2, ..., pclN)
    transposes from yolov5 (1, 1, nboxes, detection_size) to (1, 1, detection_size, nboxes)
    """
    num_classes = input_array.shape[2] - 4
    input_array = np.transpose(input_array, (0, 1, 3, 2))
    output_shape = input_array.shape[:-1] + (input_array.shape[-1] + 1,) 
    output_array = np.zeros(output_shape)
    boxes = input_array[..., :4]
    combined_class_confidences = input_array[..., 4:]
    box_confidences = np.sum(combined_class_confidences, axis=-1, keepdims=True) / num_classes
    class_confidences = combined_class_confidences / box_confidences
    output_array[..., :4] = boxes  # Copy boxes
    output_array[..., 4:5] = box_confidences  # Insert box confidence
    output_array[..., 5:] = class_confidences  # Insert separate class confidences
    
    return output_array
    
 
def process_results_letterbox(results, conf_thresh, iou_thresh, class_map, rescale_dims, original_image_shape):
    
    # i = 20
    assert CONFIG.yolo_version in ["v5", "v8"]
    if CONFIG.yolo_version == "v5":
        results = results
    elif CONFIG.yolo_version == "v8":
        results = convert_v8_to_v5(results)
    
    predictions = filter_by_confidence(results, confidence_threshold=conf_thresh)
    
    predictions = apply_class_specific_non_max_suppression(predictions, iou_threshold=iou_thresh)
    oh = original_image_shape[0]
    ow = original_image_shape[1]

    letterbox_pad = ((ow - oh) / ow / 2)
    detections = []
    for pred in predictions:
        box = pred[:4].tolist()
        x, y, w, h = box
        rescale_x = 1 / rescale_dims[0]
        rescale_y = 1 / rescale_dims[1]
        x, w = x * rescale_x, w * rescale_x
        y, h = y * rescale_y, h * rescale_y
        
        # letterbox padding
        original_x = x 
        original_y = (y - letterbox_pad) * (ow / oh)
        original_w = w 
        original_h = h  * (ow / oh)

        class_index = int(np.argmax(pred[5:]))
        confidence = float(pred[4]) * float(pred[5 + class_index])
        class_name = class_map[class_index]
        detections.append({"box": {'x': original_x, 'y': original_y, 'w': original_w, "h": original_h}, "confidence": confidence, "class_index": class_index, "class_name": class_name, "is_border": is_edge(x,y,w,h), "confidence": confidence, "box_confidence": float(pred[4]), "class_confidence": float(pred[5 + class_index])})
        # print(detections[-1])
    
    return detections


def predict(sess, image_dir, class_map):
    if CONFIG.letterboxing:
        return predict_letterbox(sess, image_dir, class_map)
    else:
        return predict_regular(sess, image_dir, class_map)
    

def predict_letterbox(sess, image_dir, class_map):
    input_shape = sess.get_inputs()[0].shape
    image = read_image_from_file(image_dir)
    image, original_image_shape = preprocess_image_letterbox(image, input_shape)
    input_data = np.array(image, dtype=np.float32)
    res = sess.run(None, {sess.get_inputs()[0].name: input_data})
    results = np.array(res)
    return process_results_letterbox(
        results, 
        conf_thresh=CONFIG.confidence_threshold, 
        iou_thresh=CONFIG.iou_threshold,
        class_map=class_map,
        rescale_dims=(image.shape[-2], image.shape[-1]),
        original_image_shape=original_image_shape)
    

def predict_regular(sess, image_dir, class_map):
    input_shape = sess.get_inputs()[0].shape
    image = read_image_from_file(image_dir)
    image = preprocess_image(image, input_shape)
    input_data = np.array(image, dtype=np.float32)
    res = sess.run(None, {sess.get_inputs()[0].name: input_data})
    results = np.array(res)
    return process_results(
        results, 
        conf_thresh=CONFIG.confidence_threshold, 
        iou_thresh=CONFIG.iou_threshold,
        class_map=class_map,
        rescale_dims=(image.shape[-2], image.shape[-1]))


if __name__ == "__main__":

    MODEL_DIR = "coco.onnx"
    IMAGE_DIR = "test_image.jpg"
    CONFIDENCE_THRESHOLD = CONFIG.confidence_threshold
    IOU_THRESHOLD = CONFIG.iou_threshold

    sess = rt.InferenceSession(MODEL_DIR)
    class_map = get_class_map(MODEL_DIR)

    input_shape = sess.get_inputs()[0].shape

    image = read_image_from_file(IMAGE_DIR)
    image = preprocess_image(image, input_shape)


    res = sess.run(None, {sess.get_inputs()[0].name: image})
    results = np.array(res)
    predictions = process_results(
        results, 
        conf_thresh=CONFIDENCE_THRESHOLD, 
        iou_thresh=IOU_THRESHOLD,
        class_map=class_map,
        rescale_dims=(image.shape[-2], image.shape[-1]))
    print(predictions)

