import pathlib
from dataset.base_dataset import DataSet
from config import load_config

class SimulationSet(DataSet):
    def __init__(self, directory):
        path = pathlib.Path(directory)
        self.config = load_config()
        print(f"images from: {path}")
        images = (str(x) for x in path.glob("*.png"))
        sorted_images = sorted(images)
        self.images = sorted_images
        
    def get_label(self, image_dir):
        path = pathlib.Path(image_dir)
        # label_dir = str(path.parent.parent) + "/labels/" + path.stem + ".txt"
        label_dir = image_dir.replace(".png", ".txt")
        label = self.parse_label(label_dir)
        label = [l for l in label if l['visibility'] > self.config.min_visibility]
        return label
    
    def profile_set(self):
        n_boxes = 0
        n_image_with_boxes = 0
        n_image_with_no_boxes = 0
        box_by_class = {}
        for image in self.images:
            label = self.get_label(image)
            n_boxes += len(label)
            if len(label) > 0:
                n_image_with_boxes += 1
            else:
                n_image_with_no_boxes += 1
            for box in label:
                class_name = box["class_name"]
                if class_name not in box_by_class:
                    box_by_class[class_name] = 0
                box_by_class[class_name] += 1
        print(f"Total images: {len(self.images)}")
        print(f"Total boxes: {n_boxes}")
        for class_name, n in box_by_class.items():
            print(f"{class_name}: {n}")
        print(f"Images with boxes: {n_image_with_boxes}")
        print(f"Images with no boxes: {n_image_with_no_boxes}")