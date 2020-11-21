from __future__ import print_function

import logging as log
import json
import os
import tensorflow as tf
import cv2
import numpy as np

log.basicConfig(level=log.DEBUG)

sess = None
input_w, input_h, input_c, input_n = (300, 300, 3, 1)

# Replace your own target label here
label_id_map = {
    1: "pedestrian"
}


def init():
    """Initialize model

    Returns: model

    """
    model_pb_path = "/usr/local/ev_sdk/model/ssd_inception_v2.pb"
    if not os.path.isfile(model_pb_path):
        log.error(f'{model_pb_path} does not exist')
        return None
    log.info('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    log.info('Initializing session...')
    global sess
    sess = tf.Session(graph=detection_graph)
    return detection_graph


def process_image(net, input_image, args=None):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: optional args

    Returns: process result

    """

    # ------------------------------- Prepare input -------------------------------------
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    ih, iw, _ = input_image.shape

    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = np.expand_dims(input_image, axis=0)

    # --------------------------- Performing inference ----------------------------------
    # Extract image tensor
    image_tensor = net.get_tensor_by_name('image_tensor:0')
    # Extract detection boxes, scores, classes, number of detections
    boxes = net.get_tensor_by_name('detection_boxes:0')
    scores = net.get_tensor_by_name('detection_scores:0')
    classes = net.get_tensor_by_name('detection_classes:0')
    num_detections = net.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: input_image})

    # --------------------------- Read and postprocess output ---------------------------
    scores = np.squeeze(scores)
    valid_index = len(scores[scores >= 0.5])

    boxes = np.squeeze(boxes)[:valid_index]
    boxes[:, 0] *= ih
    boxes[:, 2] *= ih
    boxes[:, 1] *= iw
    boxes[:, 3] *= iw
    boxes = boxes.astype(np.int32)
    classes = np.squeeze(classes)[:valid_index]
    scores = scores[:valid_index]

    detect_objs = []
    for k, score in enumerate(scores):
        label = np.int(classes[k])
        if label not in label_id_map:
            log.warning(f'{label} does not in {label_id_map}')
            continue
        ymin, xmin, ymax, xmax = boxes[k]
        detect_objs.append({
            'name': label_id_map[label],
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax)
        })
    return json.dumps({"objects": detect_objs})


if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('/usr/local/ev_sdk/data/dog.jpg')
    predictor = init()
    result = process_image(predictor, img)
    log.info(result)
