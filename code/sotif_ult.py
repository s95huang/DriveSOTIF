# from ultralytics import YOLO
import numpy as np
import cv2
# from PIL import Image


# Set classes and generate colors
classes = ['car', 'bus', 'truck', 'train', 'bike', 'motor', 'person', 'rider', 'traffic_sign', 'traffic_light', 'traffic_cone']
# np.random.seed(709)

def generate_bright_colors(n):
    colors = np.zeros((n, 3))
    hue_division = 180 / n
    for i in range(n):
        hue = int(hue_division * i)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        colors[i] = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR).squeeze()
    return colors


# IOU and NMS

def bbox_iou(box, boxes):
    """
    Calculate the Intersection over Union (IoU) of one box with an array of boxes or a single box.
    Box and boxes must be in (x1, y1, x2, y2) format.
    """
    box = np.array(box).reshape(1, 4)
    boxes = np.array(boxes).reshape(-1, 4)

    x1 = np.maximum(box[:, 0], boxes[:, 0])
    y1 = np.maximum(box[:, 1], boxes[:, 1])
    x2 = np.minimum(box[:, 2], boxes[:, 2])
    y2 = np.minimum(box[:, 3], boxes[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - intersection

    iou = intersection / union
    return iou.flatten()  # Make sure it returns a flat array suitable for indexing



def non_max_suppression(predictions, conf_thresh=0.25, iou_thresh=0.45):
    """
    Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    """
    # Filter boxes based on confidence score
    mask = predictions[:, 4] > conf_thresh
    filtered_predictions = predictions[mask]

    # Sort by confidence score
    sorted_indices = np.argsort(filtered_predictions[:, 4])[::-1]
    sorted_predictions = filtered_predictions[sorted_indices]

    keep = []
    while sorted_predictions.shape[0] > 0:
        current_box = sorted_predictions[0]
        keep.append(sorted_indices[0])

        if sorted_predictions.shape[0] == 1:
            break

        # Calculate IoU for the remaining boxes
        remaining_boxes = sorted_predictions[1:]
        iou_indices = bbox_iou(current_box[:4], remaining_boxes[:, :4], iou_thresh)

        # Filter out boxes with high IoU, keeping those below the threshold
        sorted_indices = sorted_indices[1:][iou_indices]  # Corrected filtering
        sorted_predictions = remaining_boxes[iou_indices]  # Apply the same filtering to boxes

    return keep

def calculate_entropy(probabilities):
    # probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return -(probabilities * np.log(probabilities) + (1 - probabilities) * np.log(1 - probabilities))

def bsas_with_exclusivity(detections, affinity_threshold):
    clusters = []
    for det in detections:
        placed = False
        for cluster in clusters:
            if all(det[-1] != existing_det[-1] for existing_det in cluster):  # Compare model indices
                if det[5] == cluster[0][5]:  # Check if the same class
                    for existing_det in cluster:
                        if bbox_iou(np.array(det[:4]), np.array(existing_det[:4])) >= affinity_threshold:
                            cluster.append(det)
                            placed = True
                            break
                    if placed:
                        break
        if not placed:
            clusters.append([det])
    return clusters





colors = generate_bright_colors(len(classes))


def compute_iou(box, boxes):
    """ Compute IoU of one box with multiple other boxes. """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    iou = intersection / (box_area + boxes_area - intersection)
    return iou

def non_maximum_suppression(boxes, iou_threshold=0.5):
    """ Apply Non-Maximum Suppression to remove overlapping boxes. """
    if boxes.size == 0:
        return np.array([])

    indices = np.argsort(-boxes[:, 4])
    boxes = boxes[indices]
    keep = []

    while boxes.shape[0] > 0:
        current_box = boxes[0]
        keep.append(indices[0])
        if boxes.shape[0] == 1:
            break

        ious = compute_iou(current_box, boxes[1:])
        boxes = boxes[1:][ious < iou_threshold]
        indices = indices[1:][ious < iou_threshold]

    return np.array(keep)

def global_non_maximum_suppression(detections, iou_threshold=0.5):
    """
    Global NMS across all model detections based on IOU.
    """
    if detections.shape[0] == 0:
        return np.array([])

    indices = np.argsort(-detections[:, 4])
    detections = detections[indices]
    keep = []

    while len(detections) > 0:
        current = detections[0]
        keep.append(current)
        if len(detections) == 1:
            break

        ious = bbox_iou(current[:4], detections[1:, :4])
        mask = ious < iou_threshold
        detections = detections[1:][mask]
        if detections.shape[0] != mask.sum():
            # When all IOUs are above the threshold, and none are kept
            break

    return np.array(keep)


def collect_corresponding_detections(final_detection, all_detections, iou_threshold=0.5):
    """ Collect all detections that have a high IOU with the final detection across all models. """
    corresponding_detections = []
    for detection in all_detections:
        if bbox_iou(final_detection[:4], detection[np.newaxis, :4]) >= iou_threshold:
            corresponding_detections.append(detection)
    return corresponding_detections

def cluster_detections(detections, iou_threshold=0.25):
    """ Cluster detections and map each NMS result to corresponding original detections. """
    # Apply global NMS
    final_detections = global_non_maximum_suppression(detections, iou_threshold)

    # Collect corresponding detections for each final detection
    results = []
    for final_det in final_detections:
        corresponding_dets = collect_corresponding_detections(final_det, detections, iou_threshold)
        results.append({
            'final_detection': final_det,
            'corresponding_detections': corresponding_dets
        })

    return results
