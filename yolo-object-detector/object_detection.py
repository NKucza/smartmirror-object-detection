#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from ctypes import *
import math
import random
import cv2
import time , threading
import json, sys, os, signal
import numpy as np

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("darknet.so", RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

array_to_image = lib.array_to_image
array_to_image.argtypes = [POINTER(c_char),c_int,c_int,c_int]
array_to_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def shutdown(self, signum):
	out_cap.release()
	#out.release()
	to_node("status", 'Shutdown: Done.')
	exit()

def to_node(type, message):
	# convert to json and print (node helper will read from stdout)
	try:
		print(json.dumps({type: message}))
	except Exception:
		pass
	# stdout has to be flushed manually to prevent delays in the node helper communication
	sys.stdout.flush()

FPS = 30.

def check_stdin():
	global FPS
	while True:
		lines = sys.stdin.readline()
		data = json.loads(lines)
		if 'FPS' in data:
			FPS = data['FPS']

if __name__ == "__main__":


	#out = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/object_detection_image sync=true wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)
	out_cap = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/object_detection_captions sync=true wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)

	#out.write(np.zeros((1920,1080,3), np.uint8))
	out_cap.write(np.zeros((1920,1080,3), np.uint8))

	to_node("status", "Object detection is starting...")


	cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR ,width=1080,height=1920,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink")
	#cap = cv2.VideoCapture(3)
	#cap.set(3,1920);
	#cap.set(4,1080);
	#cv2.namedWindow("object detection", cv2.WINDOW_NORMAL)

	configPath = "cfg/yolov3.cfg"
	weightPath = "data/yolov3.weights"
	metaPath = "data/coco.data"

	thresh = 0.3
	hier_thresh=.5
	nms=.45 
	debug= False

	BASE_DIR = os.path.dirname(__file__) + '/'
	os.chdir(BASE_DIR)

	signal.signal(signal.SIGINT, shutdown)

	to_node("status", "Object detection started...")

	found_Objects = []

	t = threading.Thread(target=check_stdin)
	t.start()
	

	net = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
	meta = load_meta(metaPath.encode("ascii"))

	c_char_p = POINTER(c_char)

	while True:

		start_time = time.time()

		ret, frame = cap.read()
		if ret is False:
			continue

		image_cap = np.zeros((1920,1080,3), np.uint8)

		#data = image.astype(numpy.int8)
		data_p = frame.ctypes.data_as(c_char_p)

		im = array_to_image(data_p, frame.shape[0],frame.shape[1],frame.shape[2])

		num = c_int(0)
		pnum = pointer(num)
	
		predict_image(net, im)
		dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)

		num = pnum[0]
		if nms:
			do_nms_sort(dets, num, meta.classes, nms)

		res = []
	
		for j in range(num):
			for i in range(meta.classes):
				if dets[j].prob[i] > 0:
					b = dets[j].bbox
					nameTag = meta.names[i]

					#cv2.circle(frame , (int(b.x), int(b.y))  , 5,(55,255,55), 5)
					#cv2.rectangle(frame,(int(b.x-b.w/2),int(b.y-b.h/2)),(int(b.x+b.w/2),int(b.y+b.h/2)) ,(55,255,55), 3)
					#cv2.putText(frame, str(nameTag), (int(b.x), int(b.y)), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(55,255,55), thickness=3)				

					cv2.circle(image_cap , (int(b.x), int(b.y))  , 5,(55,255,55), 5)
					cv2.rectangle(image_cap,(int(b.x-b.w/2),int(b.y-b.h/2)),(int(b.x+b.w/2),int(b.y+b.h/2)) ,(55,255,55), 3)
					cv2.putText(image_cap, str(nameTag), (int(b.x), int(b.y)), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(55,255,55), thickness=3)

		free_image(im)
		free_detections(dets, num)

		delta = time.time() - start_time
		if (1.0 / FPS) - delta > 0:
			time.sleep((1.0 / FPS) - delta)
			fps_cap = FPS
		else:
			fps_cap = 1. / delta

		#cv2.putText(frame, str(round(fps_cap)) + " FPS", (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)
		cv2.putText(image_cap, str(round(fps_cap)) + " FPS", (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)

		#cv2.imshow("object detection", frame)

		out_cap.write(image_cap)
	
		cv2.waitKey(33)


"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import pydarknet
from pydarknet import Detector, Image
import cv2
import sys
import json
import threading
import subprocess
import os
import signal
import numpy as np

if os.path.exists("/tmp/object_detection_image") is True:
	os.remove("/tmp/object_detection_image")
if os.path.exists("/tmp/object_detection_captions") is True:
	os.remove("/tmp/object_detection_captions")

#out = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/object_detection_image sync=true wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)
out_cap = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/object_detection_captions sync=true wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)

#out.write(np.zeros((1920,1080,3), np.uint8))
out_cap.write(np.zeros((1920,1080,3), np.uint8))


def to_node(type, message):
	# convert to json and print (node helper will read from stdout)
	try:
		print(json.dumps({type: message}))
	except Exception:
		pass
	# stdout has to be flushed manually to prevent delays in the node helper communication
	sys.stdout.flush()

to_node("status", "Object detection is starting...")

BASE_DIR = os.path.dirname(__file__) + '/'
os.chdir(BASE_DIR)

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("data/coco.data", encoding="utf-8"))

RECONITION_THRESHOLD = 0.7
FPS = 5
ALLOWED_OFFSET = 100

def closePoints(a,b):
    (ax,ay) = a
    (bx, by) = b
    if (bx < (ax + ALLOWED_OFFSET)) and (bx > (ax - ALLOWED_OFFSET)):
        if (by < (ay + ALLOWED_OFFSET)) and (by > (ay - ALLOWED_OFFSET)):
            return True
    return False

p = subprocess.Popen(['python', 'webstream.py'])


cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR ,width=1080,height=1920,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink")
#cap = cv2.VideoCapture(3)
#cap.set(3,1920);
#cap.set(4,1080);

def shutdown(self, signum):
	p.kill()
	out_cap.release()
	#out.release()
	to_node("status", 'Shutdown: Done.')
	exit()

signal.signal(signal.SIGINT, shutdown)


to_node("status", "Object detection started...")

found_Objects = []


def check_stdin():
    global FPS
    while True:
        lines = sys.stdin.readline()
        to_node("status", "Changing: " + lines)
        data = json.loads(lines)
        if 'FPS' in data:
            FPS = data['FPS']


t = threading.Thread(target=check_stdin)
t.start()
#cv2.namedWindow("object detection", cv2.WINDOW_NORMAL)

frame = np.zeros((1920,1920,3), np.uint8)

while True:
    image_cap = np.zeros((1920,1080,3), np.uint8)
    start_time = time.time()
    r, original_frame = cap.read()
    #original_frame = cv2.flip(np.rot90(original_frame,3),1)


    #print(frame.shape)
    #print(original_frame.shape)
    #frame = original_frame[0:1080, 0:1080]
    frame[0:1920,0:1080] = original_frame

    if r:
        dark_frame = Image(frame)
        results = net.detect(dark_frame)
        del dark_frame

        for i,(saved_cat,saved_score,saved_center,timeout_counter) in reversed(list(enumerate(found_Objects))):
            if timeout_counter < 1:
                object = (saved_cat,saved_score,saved_center,timeout_counter)[:]
                to_node("lost_object", {"name": str(object[0].decode("utf-8")), "confidence": str(format(object[1], '.4f')), "bounds": object[2][0]})
                found_Objects.remove((saved_cat,saved_score,saved_center,timeout_counter))
            else:
                found_Objects[i] = (saved_cat, saved_score, saved_center, timeout_counter - 1)


        for cat, score, bounds in results:
            if RECONITION_THRESHOLD > score:
                continue
            new_point = True
            x, y, w, h = bounds

            t = int(x-w/2)
            l = int(y-h/2)
            b = int(x+w/2)
            r = int(y+h/2)

            center = (int((t+b)/2),int((r+l)/2))

            for i,(saved_cat,saved_score,saved_center,timeout_counter) in enumerate(found_Objects):
                if saved_cat == cat:
                    if closePoints(saved_center[0],center):
                        new_point = False
                        saved_center.insert(0,center)
                        saved_center = saved_center[:15]
                        found_Objects[i] = (saved_cat, saved_score, saved_center, 15)
                       # for _center in saved_center:
                            #cv2.circle(original_frame, _center, 5, (255, 50, 50), thickness=4, lineType=8, shift=0)


            #cv2.rectangle(original_frame, (t,l) , (b,r) ,(50,255,50),thickness=4)
            #cv2.putText(original_frame, str(cat.decode("utf-8")) + " " + str(format(score,'.4f')), (int(x), int(y)),cv2.FONT_HERSHEY_COMPLEX,fontScale=2,color=(50, 255, 50), thickness=3)
            cv2.rectangle(image_cap, (t,l) , (b,r) ,(50,255,50),thickness=4)
            cv2.putText(image_cap, str(cat.decode("utf-8")) + " " + str(format(score,'.4f')), (int(x), int(y)),cv2.FONT_HERSHEY_COMPLEX,fontScale=2,color=(50, 255, 50), thickness=3)
			


            if new_point is True:
                found_Objects.append((cat,score,[center],15))
                to_node("detected_object", {"name": str(cat.decode("utf-8")), "confidence": str(format(score, '.4f')), "bounds": ((t, l), (b, r))})
                #cv2.circle(original_frame, center, 5, (50, 50, 255), thickness=4, lineType=8, shift=0)
                cv2.circle(image_cap, center, 5, (50, 50, 255), thickness=4, lineType=8, shift=0)

        frame_time = time.time() - start_time
        delta = (1.0 / FPS) - frame_time
        if delta > 0:
            time.sleep(delta)
            frame_time = (1.0 / FPS)

        #cv2.putText(original_frame, str((round(1.0/frame_time,1)))  + " FPS", (50, 100),cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)
        cv2.putText(image_cap, str((round(1.0/frame_time,1)))  + " FPS", (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)

	
        #out.write(original_frame)
        out_cap.write(image_cap)

        #cv2.imshow("object detection", original_frame)

        #cv2.waitKey(1)
        

       # cv2.imshow("preview", frame)

    #k = cv2.waitKey(1)
   # if k == 0xFF & ord("q"):
    #    break
t.stop()

"""
