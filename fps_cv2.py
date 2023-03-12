import cv2, time
from Recognition import recognition
import os
import datetime
import base64

cap = cv2.VideoCapture(0)

initial_time = time.time()
to_time = time.time()

set_fps = 10 # Set your desired frame rate

# Variables Used to Calculate FPS
prev_frame_time = 0 # Variables Used to Calculate FPS
new_frame_time = 0
model = recognition.load_ssd_model()
i = 0
flag = True
class_name = ''
label =''

last_count = 0
while True:
    while_running = time.time() # Keep updating time with each frame

    new_time = while_running - initial_time # If time taken is 1/fps, then read a frame

    if new_time >= 1 / set_fps:
        ret, frame = cap.read()
        if ret:
            # Calculating True FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            # print(fps)

            idxs, scores, bboxs = model.detect(frame, 0.6)
            cnt = len(idxs)
            cnt_dic = dict()
            
            for idx, score, bbox in zip(idxs, scores, bboxs):
                # class_name = recognition.class_names[idx]
                class_name = recognition.classes.get(idx, 'Unknown')
                label = '{0}({1:.2f})'.format(class_name, score)
                
                cv2.putText(frame, 'FPS : {0}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2) 
                cv2.putText(frame, 'cnt : {0}'.format(cnt), (200, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2) 
            
                cv2.rectangle(frame, bbox, color=(255, 0, 0), thickness=2)
                cv2.putText(frame, label, (bbox[0]+5, bbox[1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                
            if class_name == 'bird':
                cnt_dic['bird'] = cnt_dic.get('bird', 0) + 1

                if cnt_dic.get('bird') != last_count:
                    cv2.putText(frame, 'b_cnt : {0}'.format(cnt_dic['bird']), (400, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2) 

                    retval , buffer = cv2.imencode('.png', frame)
                    jpg_as_text = base64.b64encode(buffer)
                    
                    # cv2.imwrite('bird_images/bird_{}.png'.format(i), frame)
                    
                    i += 1
                    print(len(jpg_as_text.decode('utf-8')))

                    
                    
                    last_count = cnt_dic.get('bird')

            else:
                last_count = 0

            cv2.imshow('joined', frame)
            initial_time = while_running # Update the initial time with current time

        else:
            total_time_of_video = while_running - to_time # To get the total time of the video
            print(total_time_of_video)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            files = os.listdir('bird_images/')
            for file in files:
                os.remove('bird_images/{}'.format(file))

            break

cap.release()
cv2.destroyAllWindows()