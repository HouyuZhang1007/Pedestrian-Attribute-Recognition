# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys
import time

def add_path(path):
	if path not in sys.path:
		sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add project to PYTHONPATH
project_path = osp.join(this_dir)
add_path(project_path)

# Add caffe to PYTHONPATH
caffe_path = osp.join(project_path, 'caffe', 'python')
add_path(caffe_path)

'''
# Add lib to PYTHONPATH
lib_path = osp.join(project_path, 'lib')
add_path(lib_path)
'''
#os.chdir(project_path)

import caffe
import numpy as np
import cv2
import math
import scipy.io as sio
from PIL import Image
import scipy.misc


class AttributeRec:

	def __init__(self, gpu_id, model = 'VESPA_PA100K_26'):
		self.model = model
		if gpu_id == -1:
			caffe.set_mode_cpu()
    		else:
			caffe.set_mode_gpu()
        		caffe.set_device(gpu_id)

		if model == 'VESPA_PA100K_26':
			self.net = caffe.Net('./models/VesPA_PA100K_26/deploy_pa100k.prototxt', './models/VesPA_PA100K_26/vespa-pa100k_iter_35000.caffemodel', caffe.TEST)
			self.data = sio.loadmat('./PA100K_annotation.mat')
			self.name = self.data['attributes']
			self.attr_group = [range(1, 4), range(4, 7), range(13, 15), range(15, 18)]
			self.threshold = np.ones(26) * 0.5
			self.mean = self.load_mean('models/VesPA_RAP_51/rap_mean.binaryproto')[0,:]
			self.mean = self.mean.transpose((1,2,0)) #c, h, w change to h, w, c

		elif model == 'VESPA_RAP_51':
			self.net = caffe.Net('./models/VesPA_RAP_51/deploy_rap.prototxt', './models/VesPA_RAP_51/vespa-rap_iter_16000.caffemodel', caffe.TEST)
			self.data = sio.loadmat('./RAP_annotation.mat')['RAP_annotation']
			self.name = self.data['attribute_exp'][0][0]
			self.attr_group = [range(1, 4), range(4, 7), range(7, 9), range(9, 11), range(30, 36), ]
			self.threshold = np.ones(51) * 0.5
			self.mean = self.load_mean('./models/VesPA_RAP_51/rap_mean.binaryproto')[0,:]
			#self.mean = self.mean[:,:,::-1] #change to RGB
			self.mean = self.mean.transpose((1,2,0)) #c, h, w change to h, w, c
			
		elif model == 'VESPA_RAP_18':
			self.net = caffe.Net('./models/VesPA_RAP_18/deploy_rap.prototxt', './models/VesPA_RAP_18/vespa-rap18_iter_60000.caffemodel', caffe.TEST)
			self.data = sio.loadmat('./RAP_annotation.mat')['RAP_annotation']
			self.name = self.data['attribute_exp'][0][0][[0,1,7,8,9,11,12,13,17,23,24,25,26,27,28,32,35,36]]
			self.attr_group = [range(11, 14)]
			self.threshold = np.ones(18) * 0.5
			self.mean = self.load_mean('models/VesPA_RAP_51/rap_mean.binaryproto')[0,:]
			self.mean = self.mean.transpose((1,2,0)) #c, h, w change to h, w, c



		elif model == 'WPAL_PA100K_26':
			self.net = caffe.Net('./models/WPAL_GOOGLENET_PA100K_26/test_net.prototxt', './models/WPAL_GOOGLENET_PA100K_26/googlenet_spp_PA-100K_iter_170000.caffemodel', caffe.TEST)
			self.data = sio.loadmat('./PA100K_annotation.mat')
			self.name = self.data['attributes']
			self.attr_group = [range(1, 4), range(4, 7), range(13, 15), range(15, 18)]
			self.threshold = np.ones(26) * 0.5
	
			
	
	def load_mean(self, fname_bp):
    		"""
    		Load mean.binaryproto file.
    		"""
    		blob = caffe.proto.caffe_pb2.BlobProto()
    		data = open(fname_bp , 'rb' ).read()
    		blob.ParseFromString(data)
    		return np.array(caffe.io.blobproto_to_array(blob))

	def attr_rec(self,img):
		if len(img) == 0:
			print "No image attribute recognition"
			attr = -1
			orig_pred = -1
			return attr, orig_pred
		threshold = np.ones(26) * 0.5
		#threshold[8:11] = 0.7
		threshold[9:11] = 1
		'''
		imgname = []
		for i in range(len(img)):
			image = np.asarray(Image.open(img[i]))
			image =  image[:,:,0:3]
			imgname.append(image)
		'''
		
		attr, orig_pred = self.recognizer_attr(img)
		
		return attr, orig_pred



	def recognizer_attr(self, img, neglect = False):
		blobs = self._get_blobs(img, neglect)
		self.net.blobs['data'].reshape(*(blobs['data'].shape))
	
		forward_kwargs = {'data': blobs['data'].astype(np.float32, copy = False)}

		blobs_out = self.net.forward(**forward_kwargs)
		
		#pred = np.average(blobs_out['pred'], axis = 0)
		if self.model == 'VESPA_RAP_51':
			pred = blobs_out['prob-attr']
			'''
			for layer_name, blob in net.blobs.iteritems():
    				print layer_name + '\t' + str(blob.data.shape)
			'''
			orig_pred = blobs_out['prob-attr'].copy()

		if self.model == 'WPAL_PA100K_26':
			pred = blobs_out['pred']
			orig_pred = blobs_out['pred'].copy()

		if self.model == 'VESPA_PA100K_26' or self.model == 'VESPA_RAP_51' or self.model == 'VESPA_RAP_18':
			pred = blobs_out['prob-attr']
			orig_pred = blobs_out['prob-attr'].copy()


		for group in self.attr_group:
			pred = self._attr_group_norm(pred, group)
		
		if self.threshold is not None:
			for j in range(pred.shape[0]):
	        		for i in range(pred.shape[1]):
	            			pred[j][i] = 0 if pred[j][i] < self.threshold[i] else 1
		
		return pred, orig_pred
		
	def _get_blobs(self, img, neglect):
		"""Convert an image into network inputs."""
    		blobs = {'data': None}
    		blobs['data'] = self._get_image_blob(img, neglect)
    		return blobs

	def _get_image_blob(self, img, neglect):
		processed_images = []

		if self.model == 'VESPA_RAP_51' or self.model == 'VESPA_PA100K_26' or self.model == 'VESPA_RAP_18':
			for i in range(len(img)):
	    			img_ = scipy.misc.imresize(img[i], (256,256), interp='bicubic')
				#print img_.shape, "wulala"
				img_ = np.subtract(img_, self.mean)
				img_ = img_[15:15+227, 15:15+227, :]
				#img_ = img_[:,:,::-1]	#BGR to RGB
				img_ = img_.transpose((2,0,1)) #h, w, c to c, h, w
				processed_images.append(img_)


		if self.model == 'WPAL_PA100K_26':
			PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
			MIN_SIZE = 96
			TEST_SCALE = 448
			TEST_MAX_AREA = 114688
			TRAIN_RGB_JIT = 16

			for i in range(len(img)):
	    			img_orig = img[i].astype(np.float32, copy=True)
	    			img_orig -= PIXEL_MEANS
	
	    			img_shape = img_orig.shape
	    			img_size_min = np.min(img_shape[0:2])
	    			img_size_max = np.max(img_shape[0:2])
				#print img_shape[0:2],img_size_min,img_size_max
	    		
	
	    			target_size = TEST_SCALE
	    			img_scale = float(target_size) / float(img_size_max)
		
	    			# Prevent the shorter sides from being less than MIN_SIZE
	    			if np.round(img_scale * img_size_min < MIN_SIZE):
	        			img_scale = float(MIN_SIZE + 1) / img_size_min
		
	    			# Prevent the area from being larger than MAX_SIZE
	    			if np.round(img_scale * img_size_min * img_scale * img_size_max) > TEST_MAX_AREA:
	        			if neglect:
	        		    		raise ResizedImageTooLargeException
	        			img_scale = math.sqrt(float(TEST_MAX_AREA) / float(img_size_min * img_size_max))
		
	    			if img_scale * img_size_min < 64:
	        			raise ResizedSideTooShortException
		
	    			img_ = cv2.resize(img_orig, None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
				processed_images.append(img_)
	


		
	    	# Create a blob to hold the input images
	    	blob = self.img_list_to_blob(processed_images)
		
	    	return blob

	def img_list_to_blob(self, images):
		"""Convert a list of images into a network input.
    		Assumes images are already prepared (means subtracted, BGR order, ...).
    		"""
    		max_shape = np.array([img.shape for img in images]).max(axis=0)
		
		num_images = len(images)
		blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]), dtype=np.float32)
		
		
	    	
		for i in xrange(num_images):
        		img = images[i]
        		blob[i, 0:img.shape[0], 0:img.shape[1], :] = img
		# Move channels (axis 3) to axis 1
    		# Axis order will become: (batch elem, channel, height, width)
		
		if self.model == 'WPAL_PA100K_26':
			channel_swap = (0, 3, 1, 2)
			blob = blob.transpose(channel_swap)
		elif self.model == 'VESPA_RAP_51' or self.model == 'VESPA_PA100K_26' or self.model == 'VESPA_RAP_18':
			None

		return blob

	def _attr_group_norm(self, pred, group):
		for j in range(pred.shape[0]):
    			for i in group:
        			pred[j][i] = pred[j][i] if pred[j][i] == max(pred[j][group]) else 0
    		return pred

	def display(self, attr, orig_pred):
		
		
		for j in range(attr.shape[0]):
			print "--------------------PICTURE", j + 1, "--------------------"
			for i in range(attr.shape[1]):
				print ('%-23s' % list(self.name[i][0]), attr[j][i], orig_pred[j][i])
	
	def show_attr(self, attr):
		if type(attr) == int:
			return attr
		attr_image_list = []
		attr_num = len(attr[0])
		
		if self.model == 'VESPA_PA100K_26' or self.model == 'WPAL_PA100K_26':
			for i in range(len(attr)):
				attr_list = []
				for j in range(attr_num):
					if attr[i][j] == 1:
						attr_list.append(str(self.name[j][0]))
					else:
						attr_list.append(None)
	
				if attr_list[0] == None:
					attr_list[0] = 'male'
				
				attr_list_ = []
				for j in attr_list:
					if j != None:
						attr_list_.append(j)
				attr_image_list.append(attr_list_)
		
		if self.model == 'VESPA_RAP_51':	
			for i in range(len(attr)):
				attr_list = []
				for j in range(attr_num):
					if attr[i][j] == 1:
						attr_list.append(str(self.name[j][0]))
					else:
						attr_list.append(None)
	
				if attr_list[0] == None:
					attr_list[0] = 'Male'
				if attr_list[23] == None:
					attr_list[23] = 'LongSleeve'
				
				attr_list_ = []
				for j in attr_list:
					if j != None:
						attr_list_.append(j)
				attr_image_list.append(attr_list_)

		if self.model == 'VESPA_RAP_18':	
			for i in range(len(attr)):
				attr_list = []
				for j in range(attr_num):
					if attr[i][j] == 1:
						attr_list.append(str(self.name[j][0]))
					else:
						attr_list.append(None)
	
				if attr_list[0] == None:
					attr_list[0] = 'Male'
				if attr_list[1] == None:
					attr_list[1] = 'Age>18'
				if attr_list[9] == None:
					attr_list[9] = 'LongSleeve'
				
				attr_list_ = []
				for j in attr_list:
					if j != None:
						attr_list_.append(j)
				attr_image_list.append(attr_list_)
		
		return attr_image_list


if __name__ == '__main__':
	imgname = []
	imgname.append("./image/1.jpg")
	imgname.append("./image/query3.png")
	imgname.append("./image/0648_c1s3_063026_01.jpg")
	imgname.append("./image/048_0_1.bmp")
	img = []
	for i in range(len(imgname)):
		a = cv2.imread(imgname[i])
		img.append(a)
	device_mode = -1 #-1 for cpu or n for gpu device
	a = AttributeRec(device_mode, model = 'VESPA_RAP_51')
	attr, orig_pred = a.attr_rec(img)
	a.display(attr, orig_pred)
	

	
