
from __future__ import absolute_import, division, print_function, unicode_literals
import time
import cv2
import mss
import numpy as np
import os
import sys
import pyautogui
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

tf.logging.set_verbosity(tf.logging.INFO)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=17004)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)





import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))




title = "FPS benchmark"

start_time = time.time()

display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
width = 640
height = 480

monitor = {"top": 40, "left": 0, "width": width, "height": height}




from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util




PATH_TO_FROZEN_GRAPH = 'C:\\Users\\shubh\\OneDrive\\Desktop\\CSGOaimbot\\CSGO_inference_graph\\frozen_inference_graph.pb'
PATH_TO_LABELS = 'C:\\Users\\shubh\\OneDrive\\Desktop\\CSGOaimbot\\CSGO_training\\labelmap.pbtxt'
NUM_CLASSES = 4

# In[7]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# In[8]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[ ]


def Shoot(mid_x, mid_y):
    x = int(mid_x * width)
    y = int(mid_y * height + height / 9)
    #print(x,y)
    #pyautogui.dragTo(x, y )
    #pyautogui.click()


# In[ ]:


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# In[ ]:


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            image_np = np.array(sct.grab(monitor))
            # To get real color we do this:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Visualization of the results of a detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)

            array_ch = []
            array_c = []
            array_th = []
            array_t = []
            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 2:  # ch
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        array_ch.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (0, 0, 255), -1)
                if classes[0][i] == 1:  # c
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0]) / 6
                        array_c.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (50, 150, 255), -1)
                if classes[0][i] == 4:  # th
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        array_th.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (0, 0, 255), -1)
                if classes[0][i] == 3:  # t
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0]) / 6
                        array_t.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (50, 150, 255), -1)

            team = "t"
            if team == "c":
                if len(array_ch) > 0:
                    Shoot(array_ch[0][0], array_ch[0][1])
                if len(array_ch) == 0 and len(array_c) > 0:
                    Shoot(array_c[0][0], array_c[0][1])
            if team == "t":
                if len(array_th) > 0:
                    Shoot(array_th[0][0], array_th[0][1])
                if len(array_th) == 0 and len(array_t) > 0:
                    Shoot(array_t[0][0], array_t[0][1])

            # Show image with detection
            cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            # Bellow we calculate our FPS
            fps += 1
            TIME = time.time() - start_time
            if (TIME) >= display_time:
                print("FPS: ", fps / (TIME))
                fps = 0
                start_time = time.time()
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

# In[ ]:


# In[ ]:
