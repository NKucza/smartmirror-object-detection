#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from ctypes import *
import math
import random
import cv2
import time , threading
import json, sys, os, signal
import numpy as np

if os.path.exists("/tmp/object_detection_captions") is True:
	os.remove("/tmp/object_detection_captions")

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

	if os.path.exists("/tmp/object_detection_captions") is True:
		os.remove("/tmp/object_detection_captions")

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

	cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR ,width=1080,height=1920,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)
	#cap = cv2.VideoCapture(2)
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
					cv2.putText(image_cap, nameTag.decode('utf-8') , (int(b.x), int(b.y)), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(55,255,55), thickness=3)

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
	
		#cv2.waitKey(33)


