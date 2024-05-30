"""A demo to classify Raspberry Pi camera stream."""
import argparse
import collections
from collections import deque
import common
import io
import numpy as np
import operator
import os,cv2
#import picamera
import tflite_runtime.interpreter as tflite
import time

from picamera2 import Picamera2, Preview

#picam2 = Picamera2()

Category = collections.namedtuple('Category', ['id', 'score'])

def get_output(interpreter, top_k, score_threshold):
    """Returns no more than top_k categories with score >= score_threshold."""
    scores = common.output_tensor(interpreter, 0)
    categories = [
        Category(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(categories, key=operator.itemgetter(1), reverse=True)

def main():
    default_model_dir = '../all_models'
    #default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    #default_labels = 'imagenet_labels.txt'
    
    default_model = 'inception_v4_299_quant_edgetpu.tflite'
    default_labels = 'imagenet_labels.txt'
    
    #default_model = '1.tflite'
    #default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()

    #with Picamera2() as camera:
    #camera.resolution = (640, 480)
    #camera.framerate = 30
    camera = Picamera2()
    
    preview_config = camera.create_preview_configuration(main={"size": (1280, 720)})
    camera.configure(preview_config)
    #camera.start_preview(Preview.QTGL)        
    #camera.annotate_text_size = 20
    width, height, channels = common.input_image_size(interpreter)
    print(width, height, channels)
    
    camera.start()
    #time.sleep(2)
    try:
        #stream = io.BytesIO()
        #fps = deque(maxlen=20)
        #fps.append(time.time())
        while 1:
            np_array=camera.capture_array()
            np_array = np_array[:,:,:3]
            image = cv2.resize(np_array, (width, height))
            
            input = np.expand_dims(image, axis=0)
            #print(foo.size)
#             for foo in camera.capture_continuous(stream,
#                                                  format='rgb',
#                                                  use_video_port=True,
#                                                  resize=(width, height)):
            #stream.truncate()
            #stream.seek(0)
            #input = np.frombuffer(foo.getvalue(), dtype=np.uint8)
            start_ms = time.time()
            common.input_tensor(interpreter)[:,:] = np.reshape(input, common.input_image_size(interpreter))
            interpreter.invoke()
            results = get_output(interpreter, top_k=3, score_threshold=0)
            inference_ms = (time.time() - start_ms)*1000.0
            #fps.append(time.time())
            #fps_ms = len(fps)/(fps[-1] - fps[0])
            #camera.annotate_text = 'Inference: {:5.2f}ms FPS: {:3.1f}'.format(inference_ms, fps_ms)
            for result in results:
                print(100*result[1], labels[result[0]])
            print(f'Inference: {inference_ms}ms')
               #camera.annotate_text += '\n{:.0f}% {}'.format(100*result[1], labels[result[0]])
            #print(camera.annotate_text)
    except Exception as  e:
        print(e)
    finally:
        camera.stop_preview()
        camera.close()


if __name__ == '__main__':
    main()

