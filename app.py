from urllib import response
from flask import Flask, redirect, url_for, render_template, request, session, Response, redirect
import time
import datetime

from matplotlib.pyplot import text
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

app=Flask(__name__)
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2

CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-41')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

import cv2 
import numpy as np

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


sentence = []

# used to record the time when we processed last frame

 
# used to record the time at which we processed current frame
# def __del__(self):
#         try: 
#             self.cap.stop()
#             self.cap.stream.release()
#         except:
#             print('probably there\'s no cap yet :(')
#         cv2.destroyAllWindows()

def generate_frames():
    # fps_start_time = datetime.datetime.now()
    # fps = 0
    # total_frames = 0
    # dwell_time = dict()
    t = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # fps_end_time = time.time()
        # time_diff = fps_end_time - fps_start_time
        # fps = 1/(time_diff)
        # fps_start_time = fps_end_time

        # fps_text = "FPS : {:.2f}".format(fps)
        # dtime = datetime.datetime.now()
        # curr_time = datetime.datetime.now()
        # old_time = dtime
        # time_diff = curr_time - old_time
        # dtime = datetime.datetime.now()
        # sec = time_diff.total_seconds
        # dwell_time += sec
        # fps_start_time = datetime.datetime.now()
        # sec = fps_start_time.second

        # print(sec)
        # # text = "{}".format(dwell_time)
        # # text = "TIME : {}".format(fps_start_time)
        # # cv2.putText(frame, text, (5, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
        
        


        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=1,
                    min_score_thresh=.5,
                    agnostic_mode=False)
                    
        frame = cv2.resize(image_np_with_detections, (800, 600))
        word = category_index[detections['detection_classes'][np.argmax(detections['detection_scores'])]+1]['name']
        for x in detections['detection_scores']:
                if x >= 0.9:
                    cv2.putText(frame, word, (5, 75), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
        displayedSentence = ""
        time.sleep(.7)
        t += 1
        x = 0
        if t == 10:
            sentence.append(word)
            for x in detections['detection_scores']:
                if x >= 0.9:
                    for x in sentence:
                        # if x in displayedSentence:
                        #     continue
                        displayedSentence = displayedSentence + x
        text = "{}".format(displayedSentence)
        cv2.putText(frame, text, (5, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
        if t >= 10:
            t=0
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/back')
def back():
    cap.release()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, port=5001)
    