from abc import ABC, abstractmethod
from data_processing import preprocess_image, read_image_from_file, process_results, get_class_map, compute_iou_single, prediction_to_xyxy_box

from config import load_config
CONFIG = load_config()

class DataSet(ABC):
    CLASS_MAP = get_class_map(CONFIG.model_dir)
    
    @abstractmethod
    def __init__(self, directory):
        self.images = None
    
    def parse_label(self, label_dir):
        with open(label_dir, "r") as f:
            label = f.readlines()
            boxes = []
            for line in label:
                line = line.strip()
                if line == "":
                    continue
                line = line.split(" ")
                class_id = int(float(line[0]))
                x = float(line[1])
                y = float(line[2])
                w = float(line[3])
                h = float(line[4])
                visibility = float(line[5]) if len(line) > 5 else 0.0
                boxes.append({
                    "class_index": class_id,
                    "class_name": self.CLASS_MAP[class_id],
                    "box":{
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    },
                    "visibility": visibility,
                })
            return boxes
    
    @abstractmethod
    def get_label(self, image_dir):
        pass
    
    def __getitem__(self, idx):
        img = self.images[idx]
        labels = self.get_label(img)
        return img, labels
    
    def __len__(self):
        return len(self.images)
    
    
    
    
    
    

def get_data_set(directory):
    path = pathlib.Path(directory)
    print(f"images from: {path}")
    images = (str(x) for x in path.glob("images/*.jpeg"))
    sorted_images = sorted(images)
    return sorted_images


def parse_label(label_dir):
    with open(label_dir, "r") as f:
        label = f.readlines()
    boxes = []
    for line in label:
        line = line.strip()
        if line == "":
            continue
        line = line.split(" ")
        class_id = int(float(line[0]))
        x = float(line[1])
        y = float(line[2])
        w = float(line[3])
        h = float(line[4])
        boxes.append({
            "class_index": class_id,
            "class_name": CLASS_MAP[class_id],
            "box":{
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        })
    return boxes


def get_label(image_dir):
    path = pathlib.Path(image_dir)
    label_dir = str(path.parent.parent) + "/labels/" + path.stem + ".txt"
    label = parse_label(label_dir)
    return label