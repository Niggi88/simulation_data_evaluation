import pathlib
from dataset.base_dataset import DataSet

class TrainSet(DataSet):
    def __init__(self, directory):
        path = pathlib.Path(directory)
        print(f"images from: {path}")
        images = (str(x) for x in path.glob("images/*.jpeg"))
        sorted_images = sorted(images)
        self.images = sorted_images
        
    def get_label(self, image_dir):
        path = pathlib.Path(image_dir)
        label_dir = str(path.parent.parent) + "/labels/" + path.stem + ".txt"
        label = self.parse_label(label_dir)
        return label