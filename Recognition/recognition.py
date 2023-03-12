import cv2
import os

base_dir = 'C:/Users/karna/Documents/MyWorkSpace/Projects/ECS-MAIN/'

ssd_cfg_file = os.path.join(base_dir, 'dnn_model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
ssd_weights  = os.path.join(base_dir, 'dnn_model/frozen_inference_graph.pb')

yolo_cfg_file = os.path.join(base_dir, 'yolo_model/yolov4-tiny.cfg')
yolo_weights  = os.path.join(base_dir, 'yolo_model/yolov4-tiny.weights')
yolo_labels = os.path.join(base_dir, 'yolo_model/labels.txt')

yolo_v3_cfg_file = os.path.join(base_dir, 'yolov3_model/yolov3-tiny.cfg')
yolo_v3_weights = os.path.join(base_dir, 'yolov3_model/yolov3-tiny.weights')

classes = {
    1 : 'person',
    2 : 'bicycle',
    3 : 'car',
    4 : 'motorbike',
    5 : 'aeroplane',
    6 : 'bus',
    7 : 'train',
    8 : 'truck',
    9 : 'boat',
    10 : 'traffic light',
    11 : 'fire hydrant',
    13 : 'stop sign',
    15 : 'bench',
    16 : 'bird',
    17 : 'cat',
    18 : 'dog',
    19 : 'horse',
    20 : 'sheep',
    21 : 'cow',
    22 : 'elephant',
    23 : 'bear',
    27 : 'backpack',
    28 : 'umbrella',
    32 : 'tie',
    38 : 'umbrella',
    43 : 'tennis racket',
    53 : 'apple',
    73 : 'laptop',
    74 : 'mouse',
    75 : 'remote',
    76 : 'key board',
    77 : 'cell phone',
    84 : 'book',
    87 : 'scissors',
    90 : 'toothbrush'
}

class_names = []
with open(yolo_labels, 'r') as f:
    class_names = [name.strip() for name in f.readlines()]

def load_yolo_v3_model():
    net = cv2.dnn.readNet(yolo_v3_weights, yolo_v3_cfg_file)
    model = cv2.dnn_DetectionModel(net)

    model.setInputSize(416, 416)
    model.setInputScale(1.0/ 225.0)
    model.setInputSwapRB(True)

    return model

def load_ssd_model():
    net = cv2.dnn.readNet(ssd_weights, ssd_cfg_file)
    model = cv2.dnn_DetectionModel(net)

    model.setInputSize(320, 320)
    model.setInputScale(1.0/ 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    return model

def load_yolo_model():
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg_file)

    model = cv2.dnn_DetectionModel(net)

    model.setInputSize(416, 416)
    model.setInputScale(1.0/ 225.0)
    model.setInputSwapRB(True)

    return model

def detect_using_ssd(img_path):
    img = cv2.imread(img_path)

    model = load_ssd_model() 

    idxs, scores, bboxs = model.detect(img, 0.6)

    for idx, score, bbox in zip(idxs, scores, bboxs):
        class_name = classes.get(idx, 'Unknown')
        label = '{0}({1:.2f})'.format(class_name, score)

        cv2.rectangle(img, bbox, color=(255, 0, 0), thickness=2)
        cv2.putText(img, label, (bbox[0]+5, bbox[1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    cv2.imshow(img_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_using_yolo(img_path):
    img = cv2.imread(img_path)

    model = load_yolo_model() 

    idxs, scores, bboxs = model.detect(img, 0.6)

    for idx, score, bbox in zip(idxs, scores, bboxs):
        x, y, w, h = bbox

        cv2.rectangle(img, bbox, color=(255, 0, 0), thickness=2)
        cv2.putText(img, 'label', (bbox[0]+10, bbox[1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    cv2.imshow(img_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_labels(image_path:str, conf:int, model):
    img = cv2.imread(image_path)

    idxs, scores, bboxs = model.detect(img, conf)
    for idx, score, bbox in zip(idxs, scores, bboxs):
        class_name = class_names[idx]
        label = '{0}({1:.2f})'.format(class_name, score)

        cv2.rectangle(img, bbox, color=(255, 0, 0), thickness=2)
        cv2.putText(img, label, (bbox[0]+5, bbox[1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # detect_labels('images/car_test.png', 0.6, load_yolo_model())
    # detect_using_ssd('images/birds.png')
    img_path = 'images/birds.png'
    img = cv2.imread(img_path)

    model = load_yolo_model()
    idxs, scores, bboxs = model.detect(img, 0.6)
    print(idxs, scores, bboxs)

    for idx, score, bbox in zip(idxs, scores, bboxs):
        class_name = class_names[idx]

        cv2.rectangle(img, bbox, color=(255, 0, 0), thickness=2)
        cv2.putText(img, class_name, (bbox[0]+5, bbox[1]-5), cv2.FONT_HERSHEY_PLAIN , 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__' : 
    main()