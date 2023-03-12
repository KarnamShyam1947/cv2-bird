from Recognition import recognition
import cv2
import time

cap = cv2.VideoCapture(0)


def main():
    fpsLimit = 1 # throttle limit
    startTime = time.time()
    while True:
        status, frame = cap.read()
        nowTime = time.time()

        if (int(nowTime - startTime)) > fpsLimit:
            model = recognition.load_yolo_model()
            idxs, scores, bboxs = model.detect(frame, 0.6)

            for idx, score, bbox in zip(idxs, scores, bboxs):
                # class_name = recognition.classes.get(idx, 'Unknown')
                class_name = recognition.class_names[idx]
                label = '{0}({1:.2f})'.format(class_name, score)
                print(idx, '   ', label)


                cv2.rectangle(frame, bbox, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, label, (bbox[0]+5, bbox[1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
                
                if class_name == 'bird':
                    cv2.imwrite('bird_images/image.png', frame)

            cv2.imshow('Frame', frame)

            startTime = time.time()

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

if __name__ == '__main__' : 
    main()