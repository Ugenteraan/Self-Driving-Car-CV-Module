import numpy as np
import math
import cv2
import os
import rospy
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from std_msgs.msg import String

pub = rospy.Publisher('/adas/obj', String, queue_size=100)
# rospy.init_node('obj_det', anonymous=True)
# rate = rospy.Rate(10)

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
	e_x = np.exp(x - np.max(x))
	out = e_x / e_x.sum()
	return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	left, right, top, bot, mess, max_indx, confidence = None, None, None, None, None, None, None
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	


	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		
		thresh_loc = None
		thresh_height = None
		thresh_width = None
		safestatus = None

		middle_point = w/2

		# ym_per_pix = 30 / 720  # meters per pixel in y dimension
		xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
		exception_list = ['bus', 'truck']


		if mess in exception_list:
			thresh_loc = 120
			thresh_height = 180
			thresh_width = 150
						
		else:
			thresh_loc = 80
			thresh_height = 115
			thresh_width = 115


		if int(right) + thresh_loc < middle_point or int(left) - thresh_loc > middle_point :
			safestatus = True
		else:
			safestatus = False

		if safestatus == False:

			if abs(right - left) < thresh_width or abs(top - bot) < thresh_height:
				safestatus = True

		message_json = {
			"Object": str(mess),
			"Status": str(safestatus)
		}

		stringed_msg = str(message_json)
		pub.publish(stringed_msg)
		# rate.sleep()
		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, str(safestatus), (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)
		# cv2.imwrite(img_name, imgcv)



	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
