import cv2

from config import load_config
CONFIG = load_config()

def draw_detections(image_path, detections, labels, out):
    # Read the image from the file
    image = cv2.imread(image_path)
    line_thickness = 2
    font_scale = 0.75
    font_thickness = 2
    # mask = np.zeros_like(image)
    height, width, _ = image.shape
    with open(out.replace(".png", ".txt"), 'w') as file:
        # Iterate through the detections and draw them on the image
        if len(detections) > 0:
            file.write("detections\n")
        for detection in detections:
            box = detection["box"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            file.write(f"{detection['class_index']} {box['x']} {box['y']} {box['w']} {box['h']}\n")
            # Extract the bounding box coordinates and scale them to pixel values
            x = int(box['x'] * width - box['w'] * width/ 2)
            y = int(box['y'] * height - box['h'] * height / 2 )
            w = int(box['w'] * width)
            h = int(box['h'] * height)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), line_thickness)
            
            # Draw the class name and confidence
            text = f"{class_name} ({confidence * 100:.2f}%)"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=line_thickness)
        if len(labels) > 0:
            file.write("labels\n")
        for label in labels:
            if "prediction" in label: continue
            file.write(f"{label['class_index']} {label['box']['x']} {label['box']['y']} {label['box']['w']} {label['box']['h']}\n")
            box = label["box"]
            class_name = label["class_name"]
            x = int(box['x'] * width - box['w'] * width/ 2)
            y = int(box['y'] * height - box['h'] * height / 2 )
            w = int(box['w'] * width)
            h = int(box['h'] * height)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), line_thickness)
            
            # Draw the class name and confidence
            text = f"{class_name} (>{CONFIG.confidence_threshold:.2f}%)"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness=line_thickness)
    
    cv2.imwrite(out, image)
        
    return image