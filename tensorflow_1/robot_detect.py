# Import the camera server
from cscore import CameraServer

# Import OpenCV and NumPy
import cv2
import numpy as np
import argparse
import math

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from networktables import NetworkTable, NetworkTables

frame_w = 640
frame_h = 480

def main():
    cs = CameraServer.getInstance()
    cs.enableLogging()

    args = args_parser()

    # Get a CvSink. This will capture images from the camera
    cap = cv2.VideoCapture(args.camera_id)

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = cs.putVideo("Stream", 320, 240)

    # Allocating new images is very expensive, always try to preallocate
    frame = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = cv2.getTickCount()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, inference_size)
        run_inference(interpreter, frame_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]

        prev_blue = 0.0
        prev_red = 0.0
        closest_blue = [0, 0, 0, 0]
        closest_red = [0, 0, 0, 0]

        sd = NetworkTables.getTable("Ball Detector")

        for obj in objs:
            upper, lower, label = get_detection_info(inference_size, obj, labels)
            draw_boxes(frame, upper, lower, label)

            x1, y1 = lower
            x2, y2 = upper
            size = math.hypot((x2 - x1), (y2 - y1))

            if prev_blue < size and 'blue' in label:
                prev_blue = size
                closest_blue = format_info(upper, lower, label)

            if prev_red < size and 'red' in label:
                prev_red = size
                closest_red = format_info(upper, lower, label)

            if args.print:
                print(upper, lower, label)

        
        print(closest_blue)
        print(closest_red)
        
        sd.putNumberArray('closest blue', closest_blue)
        sd.putNumberArray('closest red', closest_red)

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Debugging Boxes
        # cv2.line(frame, (int(frame_w / 2), 0), (int(frame_w / 2), frame_h), (0, 255, 0))
        # cv2.line(frame, (0, int(frame_h / 2)), (frame_w, int(frame_h / 2)), (0, 255, 0))
        # cv2.line(frame, (int(3 * frame_w / 4), 0), (int(3 * frame_w / 4), frame_h), (255, 255, 0))
        # cv2.line(frame, (int(frame_w / 4), 0), (int(frame_w / 4), frame_h), (255, 255, 0))
        # cv2.line(frame, (0, int(3 * frame_h / 4)), (frame_w, int(3 * frame_h / 4)), (255, 255, 0))
        # cv2.line(frame, (0, int(frame_h / 4)), (frame_w, int(frame_h / 4)), (255, 255, 0))

        end = cv2.getTickCount()
        time = (end - start) / freq
        frame_rate_calc = 1 / time
        
        outputStream.putFrame(frame)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

def get_detection_info(inference_size, obj, labels):
    inf_w, inf_h = inference_size
    x_scalar, y_scalar = frame_w / inf_w, frame_h / inf_h

    bbox = obj.bbox.scale(x_scalar, y_scalar)
    upper_left = (int(bbox.xmin), int(bbox.ymin))
    lower_right = (int(bbox.xmax), int(bbox.ymax))

    confidence = round(obj.score, 2)
    class_name = labels.get(obj.id)
    label = '{} {}'.format(confidence, class_name)

    return upper_left, lower_right, label

def draw_boxes(frame, upper, lower, label):
    color = (255, 0, 0) if 'blue' in label else (0, 0, 255)

    x0, y0 = upper
    x1, y1 = lower

    cv2.rectangle(frame, upper, lower, color, 2)
    size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2) 
    label_y = max(y0, size[1] + 10)
    cv2.rectangle(frame, (x0, label_y - size[1] - 10), (x0 + size[0], label_y + base - 10), color, cv2.FILLED)
    cv2.putText(frame, label, (x0, label_y - 7),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 

def format_info(upper, lower, label):
    x0, y0 = upper
    x1, y1 = lower

    center_x = (x1 + x0) / 2
    center_y = (y1 + y0) / 2

    center_x = center_x - (frame_w / 2)
    center_y = -(center_y - (frame_h / 2))
    confidence = label[:4]

    size = round(math.hypot((x1 - x0), (y1 - y0)), 2)
    return center_x, center_y, size, float(confidence)

def args_parser():
    default_model = 'balls_detector.tflite'
    default_labels = 'balls_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=default_model)
    parser.add_argument('--labels', help='label file path',
                        default=default_labels)
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_id', type=int, help='Index of which video source to use. ', default=1)
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='classifier score threshold')
    parser.add_argument('--print', action='store_true', default=False, help='print detections')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    NetworkTables.initialize(server='10.41.88.2')

    main()