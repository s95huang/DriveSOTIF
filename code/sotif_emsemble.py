from ultralytics import YOLO
import numpy as np
import cv2
# from PIL import Image
from sotif_ult import *
from pathlib import Path
from random_prob import random_prob

def get_results(model, image, model_index):
    # results = model(image)
    results = model.predict(image, conf = 0.3)
    # print("Results:", results)
    boxes = []
    for r in results:
        # Each box has the format (x1, y1, x2, y2, confidence, class)
        cls = r.boxes.cls.cpu().numpy().squeeze()
        conf = r.boxes.conf.cpu().numpy().squeeze()
        xyxy = r.boxes.xyxy.cpu().numpy().squeeze()

        # if only 1, then generate box 
        if len(xyxy.shape) == 1:
            xyxy = xyxy[np.newaxis, :]
            conf = conf[np.newaxis]
            cls = cls[np.newaxis]

        # add model index to the box
        box = np.concatenate((xyxy, conf[:, None], cls[:, None], np.full((cls.shape[0], 1), model_index)), axis=1)

        boxes.append(box)
    
    return np.vstack(boxes)

def process_image(cv_img):
    img = cv_img.copy()
    # here are the 5 models trained using different random seeds. you can change them to your own model paths.
    model_paths = [
        'yolov8m_ensemble_see0/train/weights/best.pt',
        'yolov8m_ensemble_see315/train/weights/best.pt',
        'yolov8m_ensemble_see415/train/weights/best.pt',
        'yolov8m_ensemble_see9/train/weights/best.pt',
        'yolov8m_ensemble_see23/train/weights/best.pt'
    ]
    models = [YOLO(path) for path in model_paths]

    results = []
    for index, model in enumerate(models):
        results.extend(get_results(model, img, index))

    if results:
        # Ensure all entries are numpy arrays of the same shape
        results = np.array([np.array(r) for r in results])
        clusters = bsas_with_exclusivity(results, 0.95)
        # print("Clusters:", clusters)
        return clusters
    return []

def display_results(img, detections, colors, classes):
    if not isinstance(detections, np.ndarray) or detections.shape[1] < 7:
        print("Invalid detections array shape or type:", type(detections))
        return
    
    add_entropy = False  # Initialize add_entropy flag
    if detections.shape[1] == 9:
        add_entropy = True

    img_display = img.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection[:4].astype(int)
        conf = detection[4]
        cls = int(detection[5])
        model_index = int(detection[6])
        
        label = f"{classes[cls]} ({conf:.3f}) [Model {model_index}]"
        
        if add_entropy:
            entropy = detection[-1]
            # Arrange the text vertically with multiple lines
            label += f"\nEntropy: {entropy:.3f}"

        color = colors[cls]
        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
        
        # Draw each line of the label on a new line
        for i, line in enumerate(label.split('\n')):
            cv2.putText(img_display, line, (x1, y1 - 10 - (i * 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image window
    cv2.imshow("Detections", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def sotif_ensemble(image_path, visual_flag=False):
    img = cv2.imread(image_path)
    clusters = process_image(img)

    for group_index, group in enumerate(clusters):
        for i in range(len(group)):
            group[i] = np.append(group[i], group_index)
        
    flat_list = [item for sublist in clusters for item in sublist]

    # check flat_list length
    if len(flat_list) == 0:
        print("No detections found in the image")
        return None, None

    detections_array = np.vstack(flat_list)

    final_detections = cluster_detections(detections_array, iou_threshold=0.4)

    final_boxes = np.array([result['final_detection'] for result in final_detections])

    # print("Final boxes:", final_boxes)

    if visual_flag:
        display_results(img, final_boxes, colors, classes)

    return final_boxes, detections_array


def calculate_entropies(output_boxes, detections_array, classes, use_avg_flag=True):
    """Calculate and append entropies for each output box based on associated cluster detections."""
    # Create a new array to store the entropy values
    entropy_values = np.zeros((output_boxes.shape[0], 1))

    for i, box in enumerate(output_boxes):
        cluster_id = int(box[7])  # Assuming the cluster ID is at index 7

        # Get all boxes in the same cluster
        cluster_boxes = detections_array[detections_array[:, 7] == cluster_id]

        # Calculate class probabilities for the cluster
        probabilities = np.zeros(len(cluster_boxes))

        # get the probabilities from each model
        for j, cluster_box in enumerate(cluster_boxes):
            probabilities[j] = cluster_box[4]

        avg = np.mean(probabilities)

        # Calculate entropy

        if use_avg_flag:
            # generate random probabilities
            random_probabilities = random_prob(avg, len(classes))

            # sum entropies
            entropy = np.max([calculate_entropy(prob) for prob in random_probabilities]) + calculate_entropy(avg)
            # print("calculate_entropy(avg):", calculate_entropy(avg), "Entropy:", entropy)
        else:
            entropy = calculate_entropy(avg)

        # print("Entropy:", entropy, 'len(cluster_boxes):', len(cluster_boxes))

        if len(cluster_boxes) < 5:
            entropy *= (1 + 0.1 * (5 - len(cluster_boxes)))

        # print("num_classes_detected:", num_classes_detected, "Entropy:", entropy)

        entropy_values[i] = entropy

    # Append entropy to the output_boxes
    output_boxes = np.hstack((output_boxes, entropy_values))

    return output_boxes






# combine these

def sotif(image_path, visual_flag=False, thresh_med=1.2, thresh_high=1.6):
    img = cv2.imread(image_path)

    if img is None:
        print("Invalid image path:", image_path)
        return None
    
    final_boxes, detections_array = sotif_ensemble(image_path, False)

    if final_boxes is None or detections_array is None:
        return None

    final_boxes_with_entropies = calculate_entropies(final_boxes, detections_array, classes)

    if visual_flag:
        display_results(img, final_boxes_with_entropies, colors, classes)

    # get columns 0 to 5 and -1
    # 0:4 are the box coordinates, 4 is the confidence, -1 is the entropy, 5 is the class, 6 is the model index, 7 is the cluster index

    # remove the model index and cluster index
    final_boxes_with_entropies = final_boxes_with_entropies[:, [0, 1, 2, 3, 4, 5, -1]]


    annotated_boxes = []
    for box in final_boxes_with_entropies:
        entropy = box[-1].round(3)
        # Determine the uncertainty level based on entropy thresholds
        if entropy < thresh_med:
            uncertainty_level = "low"
        elif entropy < thresh_high:
            uncertainty_level = "medium"
        else:
            uncertainty_level = "high"

        # add field bbox, and confidence and type
        
        bbox = box[:4]
        confidence = box[4].round(3)
        cls = box[5]
        cls = classes[int(cls)]
        
        # create an brand new list

        annotated_box = []
        annotated_box.extend(["bbox", bbox, "confidence", confidence, "type", cls, "Entropy", entropy, "uncertainty level", uncertainty_level])
        # annotated_box.extend(["bbox", bbox, "confidence", confidence, "type", cls, "Entropy", entropy, "uncertainty level", uncertainty_level])

        annotated_boxes.append(annotated_box)

    return annotated_boxes



# Test the complete setup
image_path = 'images/image.png'
final_boxes = sotif(image_path, visual_flag=True)
print("Final boxes:", final_boxes)