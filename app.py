from pathlib import Path
import argparse
import time
import numpy as np
import cv2

import torch
import torch.nn as nn

from models.experimental import Ensemble
from models.common import Conv, DWConv
from utils.general import non_max_suppression, apply_classifier

import logging
from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_TEMPLATE = "env.count"

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    parser.add_argument('--weight', type=str, default='yolov7.pt', help='model name')
    parser.add_argument('--labels', dest='labels',
                        action='store', default='coco.names', type=str,
                        help='Labels for detection')


    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')


    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="bottom",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Continuous run flag')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')


    return parser.parse_args()

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


class YOLOv7_Main():
    def __init__(self, args, weightfile):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = Ensemble()
        ckpt = torch.load(weightfile, map_location=self.device)
        self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model = self.model.half()
        self.model.eval()

        self.class_names = load_class_names(args.labels)


    def run(self, frame, args):
        sized = cv2.resize(frame, (640, 640))

        image = sized / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).half()
        image = image.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(image)[0]
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=True)

        return pred, self.class_names

def detect(yolov7_main, sample, do_sampling, plugin, args):
    frame = sample.data
    timestamp = sample.timestamp
    results, outclass = yolov7_main.run(frame, args)
    print('detection done')
    results = results[0]

    if do_sampling:
        found = {}
        for result in results:
            l = result[0] * w/640  ## x1
            t = result[1] * h/640  ## y1
            r = result[2] * w/640  ## x2
            b = result[3] * h/640  ## y2
            conf = round(result[4], 2)
            name = outclass[int(result[5])]
            frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255,0,0), 2)
            frame = cv2.putText(frame, f'{name}:{conf}', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            if not name in found:
                found[name] = 1
            else:
                found[name] += 1

        sample.data = frame
        sample.save('yolov7.jpg')
        plugin.upload_file('yolov7.jpg')
        print('saved')

        detection_stats = 'found objects: '
        for name, count in found.items():
            detection_stats += f'{name}[{cound}] '
            plugin.publish(f'{TOPIC_TEMPLATE}.{name}', count, timestamp=timestamp)
        print(detection_stats)
    else:
        found = {}
        for result in results:
            name = outclass[int(result[5])]
            if not name in found:
                found[name] = 1
            else:
                found[name] += 1

        detection_stats = 'found objects: '
        for name, count in found.items():
            detection_stats += f'{name}[{count}] '
            plugin.publish(f'{TOPIC_TEMPLATE}.{name}', count, timestamp=timestamp)
        print(detection_stats)

if __name__ == "__main__":
    print('loading args')
    args = get_arguments()
    print('loading model')
    yolov7_main = YOLOv7_Main(args, args.weight)

    sampling_countdown = -1
    if args.sampling_interval >= 0:
        sampling_countdown = args.sampling_interval

    while True:
        with Plugin() as plugin:
            with Camera(args.stream) as camera:
                sample = camera.snapshot()


            do_sampling = False
            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                do_sampling = True
                sampling_countdown = args.sampling_interval


            detect(yolov7_main, sample, do_sampling, plugin, args)
            if not args.continuous:
                exit(0)
