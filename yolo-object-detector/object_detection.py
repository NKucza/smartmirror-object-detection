#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from ctypes import *
import math
import random
import cv2
import time , threading
import json, sys, os, signal
import numpy as np
import darknet

if os.path.exists("/tmp/object_detection_captions") is True:
	os.remove("/tmp/object_detection_captions")

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

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

if __name__ == "__main__":


	#out = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/object_detection_image sync=true wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)
	out_cap = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/object_detection_captions sync=true wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)

	#out.write(np.zeros((1920,1080,3), np.uint8))
	out_cap.write(np.zeros((1920,1080,3), np.uint8))

	to_node("status", "Object detection is starting...")

	cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR ,width=1080,height=1920,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", cv2.CAP_GSTREAMER)
	#cap = cv2.VideoCapture(2)
	#cap.set(3,1920);
	#cap.set(4,1080);
	cv2.namedWindow("object detection", cv2.WINDOW_NORMAL)

	darknet.set_gpu(1)

	configPath = "cfg/yolov3.cfg"
	weightPath = "data/yolov3.weights"
	metaPath = "data/coco.data"

	thresh = 0.7
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
	
	netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
	metaMain = darknet.load_meta(metaPath.encode("ascii"))

	    # Create an image we reuse for each detect
	darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)
        

	#raster for hand tracing.. here the image resolution 
	horizontal_division = 36.0
	vertical_division =  64.0

	detectionArray = np.zeros((int(vertical_division),int(horizontal_division),metaMain.classes),dtype=np.uint8)
	
	while True:

		start_time = time.time()

		ret, frame = cap.read()
		if ret is False:
			continue

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

		image_cap = np.zeros((1920,1080,3), np.uint8)

		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

		dets = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

		detectionArray = np.where(detectionArray >0 , detectionArray -1 , 0)
	
		for det in dets:
			x, y, w, h = det[2][0],\
            det[2][1],\
            det[2][2],\
            det[2][3]

			xrel = int(x / darknet.network_width(netMain) * horizontal_division)
			yrel = int(y / darknet.network_height(netMain) * vertical_division)
			

			x = x / darknet.network_width(netMain) * 1080
			y = y / darknet.network_height(netMain) * 1920
			w = w / darknet.network_width(netMain) * 1080
			h = h / darknet.network_height(netMain) * 1920

			xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
			pt1 = (xmin, ymin)
			pt2 = (xmax, ymax)

			for j in range(metaMain.classes):
				if det[0] == metaMain.names[j]:
					i = j
					break
			
			cv2.circle(frame , (int(x), int(y))  , 5,(55,255,55), 5)
			cv2.rectangle(frame,pt1,pt2,(55,255,55), 3)
			cv2.putText(frame, det[0].decode('utf-8'), (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(55,255,55), thickness=3)

			cv2.circle(image_cap , (int(x), int(y))  , 5,(55,255,55), 5)
			cv2.rectangle(image_cap,pt1,pt2,(55,255,55), 3)
			cv2.putText(image_cap, det[0].decode('utf-8'), (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(55,255,55), thickness=3)


			detectionArray[yrel,xrel,i] += 2

			if detectionArray[yrel,xrel,i] == 2 * FPS:
				detectionArray[yrel,xrel,i] = 0			
			if detectionArray[yrel,xrel,i] == FPS:
				wrel = w / darknet.network_width(netMain)
				hrel = h / darknet.network_height(netMain)
				to_node("detected",{"name": str(det[0].decode("utf-8")), "center": (float("{0:.5f}".format(xrel/horizontal_division )),float("{0:.5f}".format(yrel/vertical_division))),"box": (float("{0:.5f}".format(hrel)),float("{0:.5f}".format(wrel)))})
		

		delta = time.time() - start_time
		if (1.0 / FPS) - delta > 0:
			time.sleep((1.0 / FPS) - delta)
			fps_cap = FPS
		else:
			fps_cap = 1. / delta

		cv2.putText(frame, str(round(fps_cap)) + " FPS", (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)
		cv2.putText(image_cap, str(round(fps_cap)) + " FPS", (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)

		cv2.imshow("object detection", frame)

		out_cap.write(image_cap)
	
		cv2.waitKey(33)


