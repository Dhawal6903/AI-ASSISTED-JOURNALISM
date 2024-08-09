from ultralytics import YOLO
import torch
import copy
from PIL import Image
import cv2
import numpy as np
# Initialize

model5 = YOLO('yolov5m.pt')
model8 = YOLO('yolov8m.pt')





from collections import defaultdict

def detect_objects_on_image(buf):
    # Load YOLOv5 and YOLOv8 models for floods, fire, and buildings
    
    # YOLOv5 models
    model5_floods = YOLO("./models/yolov5/floods.pt")
    model5_fire = YOLO("./models/yolov5/fire.pt")
    model5_buildings = YOLO("./models/yolov5/buildings.pt")
    
    # YOLOv8 models
    model8_floods = YOLO("./models/yolov8/last.pt")
    model8_fire = YOLO("./models/yolov8/fire.pt")
    model8_buildings = YOLO("./models/yolov8/buildings.pt")
    
    # Make predictions using pairs of models (YOLOv5 and YOLOv8)
    predictions_pairs = [
        (model5_floods, model8_floods),
        (model5_fire, model8_fire),
        (model5_buildings, model8_buildings)
    ]
    
    # Ensemble predictions from pairs
    ensemble_predictions = defaultdict(list)
    
    output = []
    output2 = []
    track = set()
    for model5, model8 in predictions_pairs:
        results5 = model5.predict(buf)
        results8 = model8.predict(buf)

        # print("These are the output of results")

        # print(results5[0])

        # print(results8[0])


        for box1 in results5[0].boxes:
            x1, y1, x2, y2 = [
                round(x) for x in box1.xyxy[0].tolist()
            ]
            class_id = box1.cls[0].item()
            prob = round(box1.conf[0].item(), 2)
            if class_id not in track:
                track.add(class_id)
                if(prob >= 0.5):
                    output.append([
                        x1, y1, x2, y2, results5[0].names[class_id], prob
                    ])
            output2.append([
                x1, y1, x2, y2, results5[0].names[class_id], prob
            ])
            print(x1, y1, x2, y2, results5[0].names[class_id], prob)
            print("This is the end of the first box")
        
        for box2 in results8[0].boxes:
            x1, y1, x2, y2 = [
                round(x) for x in box2.xyxy[0].tolist()
            ]
            class_id = box2.cls[0].item()
            prob = round(box2.conf[0].item(), 2)
            print(x1, y1, x2, y2, results8[0].names[class_id], prob)
            if class_id not in track:
                track.add(class_id)
                if (prob >= 0.5):
                    if('Heavy floods have occured' in results8[0].names[class_id] and prob >= 0.9):
                        output.append([
                            x1, y1, x2, y2, results8[0].names[class_id], prob
                        ])
                        output2.append([
                            x1, y1, x2, y2, results8[0].names[class_id], prob
                        ])
                        continue
                    output.append([
                        x1, y1, x2, y2, results8[0].names[class_id], prob
                    ])
            output2.append([
                x1, y1, x2, y2, results8[0].names[class_id], prob
            ])
            print("This is the end of the second box")


        # print(output) 
    print(output)
    return output, output2


def draw_boxes_on_image(image_path, bounding_boxes):
    # Read the image
    nparr = np.frombuffer(image_path.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Font settings for text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Draw bounding boxes and labels on the image
    for box in bounding_boxes:
        x1, y1, x2, y2, label, _ = box  # Extract coordinates and label from each box
        # Draw the rectangle around the object
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle, 2px thickness
        # Add label text near the bounding box
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 255), font_thickness)
    
    # Display the image with bounding boxes and labels
    # cv2.imshow('Image with Bounding Boxes', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img







# def detect_objects_on_image(buf):
#     """
#     Function receives an image,
#     passes it through YOLOv8 neural network
#     and returns an array of detected objects
#     and their bounding boxes
#     :param buf: Input image file stream
#     :return: Array of bounding boxes in format 
#     [[x1,y1,x2,y2,object_type,probability],..]
#     """
#     # yolov5 models
#     model5_floods = YOLO("./models/yolov5/floods.pt")
#     model5_fire = YOLO("./models/yolov5/fire.pt")
#     model5_buildings = YOLO("./models/yolov5/buildings.pt")




#     # yolov8 models
#     model5_floods = YOLO("./models/yolov8/floods.pt")
#     model5_fire = YOLO("./models/yolov8/floods.pt")
#     model5_buildings = YOLO("./models/yolov8/buildings.pt")




#     results = model.predict(buf)
#     result = results[0]
#     output = []
#     for box in result.boxes:
#         x1, y1, x2, y2 = [
#           round(x) for x in box.xyxy[0].tolist()
#         ]
#         class_id = box.cls[0].item()
#         prob = round(box.conf[0].item(), 2)
#         output.append([
#           x1, y1, x2, y2, result.names[class_id], prob
#         ])
#     return output
