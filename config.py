
import yaml
from pydantic import BaseModel
from typing import Union
from pathlib import Path
from pprint import pprint

class Config(BaseModel):
    model_dir: Path # Union[str, Path]
    folder: Path # Union[str, Path]
    out_dir: Path # Union[str, Path]
       
    confidence_threshold: float
    iou_threshold: float
    letterboxing: bool
    yolo_version: str
    
    
def load_config():
    with open('config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)