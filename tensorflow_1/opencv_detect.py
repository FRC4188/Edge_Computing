import argparse
import cv2

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference 

def main():
    args = args_parser()
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_id)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = cv2.getTickCount()

        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]

        for obj in objs:
            upper, lower, label = get_detection_info(frame, inference_size, obj, labels)
            draw_boxes(frame, upper, lower, label)

            if args.print:
                print(label)
        
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        end = cv2.getTickCount()
        time = (end - start) / freq
        frame_rate_calc = 1 / time

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_detection_info(cv2_im, inference_size, obj, labels):
    img_h, img_w, img_c = cv2_im.shape
    inf_w, inf_h = inference_size
    x_scalar, y_scalar = img_w / inf_w, img_h / inf_h

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
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--print', action='store_true', default=False, help='print detections')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()