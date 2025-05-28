import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check model
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# Determine source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid input: {img_source}')
    sys.exit(0)

# Resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Recorder setup
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video or USB sources.')
        sys.exit(0)
    if not user_res:
        print('Specify resolution when recording.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Initialize source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Main loop
while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Camera disconnected or video ended.')
            break
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    detected_items = []

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            detected_items.append(classname)
            object_count += 1

    # Send detected items to local server
    try:
        requests.post("http://localhost:5000/update", json={"items": detected_items})
    except Exception as e:
        print(f"Error sending data to server: {e}")

    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results', frame)
    if record: recorder.write(frame)

    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'picamera'] else 0)
    if key in [ord('q'), ord('Q')]: break
    elif key in [ord('s'), ord('S')]: cv2.waitKey()
    elif key in [ord('p'), ord('P')]: cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']: cap.release()
elif source_type == 'picamera': cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
