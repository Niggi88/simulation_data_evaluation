import os
import cv2
import numpy as np

from config.logging import get_logger

logger = get_logger()


def draw_detections(image_path, detections, outdir):
    # Read the image from the file
    image = cv2.imread(image_path)
    line_thickness = 2
    font_scale = 0.75
    font_thickness = 2
    mask = np.zeros_like(image)
    height, width, _ = image.shape
    
    # Iterate through the detections and draw them on the image
    for detection in detections:
        box = detection["box"]
        confidence = detection["confidence"]
        class_name = detection["class_name"]

        # Extract the bounding box coordinates and scale them to pixel values
        x = int(box['x'] * width - box['w'] * width/ 2)
        y = int(box['y'] * height - box['h'] * height / 2 )
        w = int(box['w'] * width)
        h = int(box['h'] * height)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)
    
        
    
    darkened_image = image * 0.3
    final_image = np.where(mask == 255, image, darkened_image).astype(np.uint8)
    final_image = image
    
    for detection in detections:
        box = detection["box"]
        confidence = detection["confidence"]
        class_name = detection["class_name"]

        # Extract the bounding box coordinates and scale them to pixel values
        x = int(box['x'] * width - box['w'] * width/ 2)
        y = int(box['y'] * height - box['h'] * height / 2 )
        w = int(box['w'] * width)
        h = int(box['h'] * height)
        
        
        # Draw the bounding box
        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), line_thickness)
        
        # Draw the class name and confidence
        text = f"{class_name} ({confidence * 100:.2f}%)"
        cv2.putText(final_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=line_thickness)
    # Optionally, save the image with detections
    out = os.path.join(os.path.dirname(outdir), os.path.basename(image_path))
    out = out.replace(".jpeg", ".png")
    print("saving to", out)
    cv2.imwrite(out, final_image)
    
    # write to file
    out_txt = out.replace(".png", ".txt")
    print("writing to file      ", out)
    print("writing to file txt: ", out_txt)
    with open(out_txt, 'w') as file:
        for detection in detections:
            box = detection["box"]
            border_detection = "unknown" if "is_border" not in detection.keys() else detection["is_border"]
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            confidence, box_confidence, class_confidence = detection["confidence"], detection["box_confidence"], detection["class_confidence"]
            
            line = ' '.join([str(detection["class_index"])] + 
                            [f"{float(x):.4f}" for x in [y-0.5*h, x-0.5*w, y+0.5*h, x+0.5*w]] + 
                            [str(border_detection)] + 
                            [str(confidence), str(box_confidence), str(class_confidence)]) + "\n"
            file.write(line)
            # if "is_border" in detection.keys() and detection["is_border"]:
            #     logger.info("border detection: "+ out_txt)