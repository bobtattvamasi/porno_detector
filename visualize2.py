#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import sys
import warnings
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
from model_nsfw import OpenNsfwModel, InputType
from threading import Thread
import threading
import time
import linecache
from logg import QTextEditLogger
import logging
from image_utils import create_yahoo_image_loader

import gc

FRAMES_DIR_NAME = 'frames'
CHANNEL_ID_KEY = 'channel_id'
SECOND_CHANNEL_ID_KEY = 'channel_id2'
NSFW_SCORE_KEY = 'nsfw_score'
NSFW_THRESHOLD = 0.5
CHNL_THRESHOLD = 0.55
NSFW_GPU_MEMORY = 0.4
CHNL_GPU_MEMORY = 0.4
UNDEFINED = -1
UNKNOWN_CHANNEL_ID = -2
INITIAL_CHANNEL_ID = -3

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def import_nsfw_model():
	model_weights = "models/retrain_nsfw.npy"
	#model_weights = "models/open_nsfw-weights.npy"

	input_type = InputType.TENSOR.name.lower()

	model = OpenNsfwModel()
	config = tf.ConfigProto()
	# config = tf.ConfigProto(intra_op_parallelism_threads=8,
	#   inter_op_parallelism_threads=8)
	config.gpu_options.per_process_gpu_memory_fraction = NSFW_GPU_MEMORY

	sess_nsfw = tf.Session(config=config)

	input_type = InputType[input_type.upper()]
	model.build(weights_path=model_weights, input_type=input_type)

	fn_load_image = create_yahoo_image_loader()
	sess_nsfw.run(tf.global_variables_initializer())

	return fn_load_image, sess_nsfw, model

def import_logo_detector_model():
	model_path = 'models/channels_frozen_inference_graph.pb'
	logo_detection_graph = tf.Graph()
	with logo_detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(model_path, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	labels = []
	labels_path = 'channels_labels.txt'  # HERE - файл с подписями для классов
	with open(labels_path) as f:
		labels = f.readlines()
	labels = [s.strip() for s in labels]
	return LogoDetector(logo_detection_graph, labels, [i+1 for i in range(17)], CHNL_THRESHOLD), labels

def raise_exception(e):
	"""Функция для выброса сообщения об ошибке с указанием где и что произошло"""
	exc_type, exc_obj, exc_tb = sys.exc_info()
	f = exc_tb.tb_frame
	fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
	line = linecache.getline(fname, exc_tb.tb_lineno, f.f_globals)
	#print(exc_type, fname, exc_tb.tb_lineno, line)
	print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(fname, exc_tb.tb_lineno, line.strip(), exc_obj))
	print(e)

class LogoDetector():
	def __init__(self, detection_graph, labels, classes_to_detect, confidence_level):
		super().__init__()      
		self.detection_graph = detection_graph      
		
		self.default_graph = self.detection_graph.as_default()
		# config = tf.ConfigProto(intra_op_parallelism_threads=5,
	 #  inter_op_parallelism_threads=5)
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = CHNL_GPU_MEMORY
		self.sess = tf.Session(graph=self.detection_graph, config=config)
		self.labels = labels

		# Definite input and output Tensors for detection_graph
		self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

		self.classes_to_detect = classes_to_detect
		self.confidence_level = confidence_level

	def process(self, frame):
		#assert(len(self.classes_to_detect) > 0 and self.confidence_level <= 1, 'WUT?')

		# Expand dimensions since the trained_model expects frames to have shape: [1, None, None, 3]
		im_height, im_width, _ = frame.shape
		rect_img = np.ones((im_height,im_width),frame.dtype)
		rect_img   = cv2.rectangle(rect_img, (int(im_width/4), int(im_height/4)), (int(3*im_width/4), int(3*im_height/4)), 0, thickness=-1)
		frame = cv2.bitwise_and(frame, frame, mask=rect_img)
		
		frame_np_expanded = np.expand_dims(frame, axis=0)

		# Actual detection.
		(boxes, scores, classes, num) = self.sess.run(
			[self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
			feed_dict={self.image_tensor: frame_np_expanded})

		im_height, im_width, _ = frame.shape
		all_boxes = []
		for i in range(boxes.shape[1]):
			all_boxes.append((int(boxes[0, i, 1] * im_width),
							  int(boxes[0, i, 0] * im_height),
							  int(boxes[0, i, 3] * im_width),
							  int(boxes[0, i, 2] * im_height)))

		all_scores = scores[0].tolist()
		all_classes = [int(x) for x in classes[0].tolist()]
		all_labels = [self.labels[int(x) - 1] for x in all_classes]

		ret_boxes = []
		ret_scores = []
		ret_classes = []
		ret_labels = []
		for i in range(len(all_boxes)):
			if all_classes[i] in self.classes_to_detect and all_scores[i] > self.confidence_level:
				ret_boxes.append(all_boxes[i])
				ret_scores.append(all_scores[i])
				ret_classes.append(all_classes[i])
				ret_labels.append(all_labels[i])

		# чем полезен int(num[0]) ?
		# print('len(all_boxes) =', len(ret_boxes), 'int(num[0]) =', int(num[0]))
		return ret_boxes, ret_scores, ret_classes, ret_labels

	def get_label(self, class_id):
		if class_id < 1 or class_id > len(self.labels):
			return 'Неизвестно'
		else:
			return self.labels[class_id - 1]

	def close(self):
		self.sess.close()
		# self.default_graph.close()  # AttributeError: '_GeneratorContextManager' object has no attribute 'close'
		

class VideoCapture(QWidget):
	frame_processed_signal = pyqtSignal(dict)

	def __init__(self, filename, frames_per_detection, buffer_size, parent):
		super(QWidget, self).__init__()
		self.timer = QTimer()
		self.filename = filename
		self.cap = cv2.VideoCapture(str(filename))
		self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
		print("self.frame_rate = = ", self.frame_rate)

		self.fn_load_image, self.sess_nsfw, self.model = import_nsfw_model()
		self.logo_detector, self.logo_labels = import_logo_detector_model()

		self.current_channel_id = INITIAL_CHANNEL_ID
		self.tv_decision = "ТВ"

		self.count_frame = 0
		self.buffer_channel = []
		self.buffer_channel2 = []
		self.buffer_nsfw = []

		sliders_widget = QWidget(self)
		grid_layout = QGridLayout()

		self.video_frame = QLabel()
		grid_layout.addWidget(self.video_frame, 0, 0, 1, 3)

		self.logTextBox = QTextEditLogger(self)
		self.logTextBox.widget.setTextColor(QColor(0,0,0))
		grid_layout.addWidget(self.logTextBox.widget, 0, 3, 5, 1)        

		channel_label = QLabel()
		channel_label.setText('Канал:')
		channel_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(channel_label, 1, 0)
		content_label = QLabel()
		content_label.setText('Контент:')
		content_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(content_label, 2, 0)
		self.channel_value_label = QLabel()
		self.channel_value_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(self.channel_value_label, 1, 1)
		self.content_value_label = QLabel()
		self.content_value_label.setText("<font color='green'>Безопасно</font>")
		self.content_value_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(self.content_value_label, 2, 1)
		self.channel_conf_label = QLabel()
		self.channel_conf_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(self.channel_conf_label, 1, 2)
		self.content_conf_label = QLabel()
		self.content_conf_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(self.content_conf_label, 2, 2)

		interval_label = QLabel()
		interval_label.setText('Интервал обработки:')
		interval_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(interval_label, 3, 0)
		buffer_size_label = QLabel()
		buffer_size_label.setText('Размер буфера:')
		buffer_size_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(buffer_size_label, 4, 0)
		interval_value_label = QLabel()
		interval_value_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(interval_value_label, 3, 2)
		buffer_size_value_label = QLabel()
		buffer_size_value_label.setFont(QFont("Courier", 20, QFont.Bold))
		grid_layout.addWidget(buffer_size_value_label, 4, 2)

		interval_slider = QSlider(Qt.Horizontal)
		def on_interval_slider_value_changed():
			self.frames_per_detection = interval_slider.value()
			interval_value_label.setText('{:>5} '.format(self.frames_per_detection))
		interval_slider.setMinimum(3)
		interval_slider.setMaximum(15)
		interval_slider.setTickInterval(1)
		interval_slider.valueChanged.connect(on_interval_slider_value_changed)
		interval_slider.setValue(7)
		grid_layout.addWidget(interval_slider, 3, 1)
		buffer_size_slider = QSlider(Qt.Horizontal)
		def on_buffer_size_slider_value_changed():
			self.buffer_size = buffer_size_slider.value()
			buffer_size_value_label.setText('{:>5} '.format(self.buffer_size))
		buffer_size_slider.setMinimum(3)
		buffer_size_slider.setMaximum(15)
		buffer_size_slider.setTickInterval(2)
		buffer_size_slider.valueChanged.connect(on_buffer_size_slider_value_changed)
		buffer_size_slider.setValue(11)
		grid_layout.addWidget(buffer_size_slider, 4, 1)

		sliders_widget.setLayout(grid_layout)
		parent.layout.addWidget(sliders_widget)

		self.actual_channel_id = UNKNOWN_CHANNEL_ID
		self.actual_channel_id2 = UNKNOWN_CHANNEL_ID
		self.previous_channel_id = UNKNOWN_CHANNEL_ID
		self.is_porn_detected = False
		self.scores = None
		self.result_dict = None
		self.thread_porn = None
		self.gc_thread = None

	def process_frame(self, frame):
		result_dict = {}
		try:
			boxes, scores_logos, classes, labels = self.logo_detector.process(frame)
			print('Detector:', classes, scores_logos)
			max_score, max_i = 0, UNDEFINED
			for i in range(len(scores_logos)):
				if scores_logos[i] > max_score:
					max_score = scores_logos[i]
					max_i = i
			max_score2, max_i2 = 0, UNDEFINED
			
				
			for i in range(len(scores_logos)):
				if scores_logos[i] > max_score2:
					if scores_logos[i] != max_score:
						max_score2 = scores_logos[i]
						max_i2 = i


			if max_i == UNDEFINED:
				result_dict.update({CHANNEL_ID_KEY: (UNKNOWN_CHANNEL_ID, 0)})
			else:
				result_dict.update({CHANNEL_ID_KEY: (classes[max_i], max_score)})

			if max_i2 == UNDEFINED:
				result_dict.update({SECOND_CHANNEL_ID_KEY: (UNKNOWN_CHANNEL_ID, 0)})
			else:
				result_dict.update({SECOND_CHANNEL_ID_KEY: (classes[max_i2], max_score2)})

		except Exception as e:
			raise_exception(e)
			#print('ои веи')
			result_dict.update({CHANNEL_ID_KEY: (UNKNOWN_CHANNEL_ID, 0)})
			result_dict.update({SECOND_CHANNEL_ID_KEY: (UNKNOWN_CHANNEL_ID, 0)})
		# if KeyboardInterrupt:
		# 	os.system("ps axf | grep vlc | grep -v grep | awk '{print \"kill -9 \" $1}' | sh")


		if not os.path.isdir(FRAMES_DIR_NAME):
			os.mkdir(FRAMES_DIR_NAME)
		img_name = '{}/image_{}.jpg'.format(FRAMES_DIR_NAME, (threading.currentThread().getName()[-1:]))
		cv2.imwrite(img_name, frame)
		frame = self.fn_load_image(img_name)
		scores = self.sess_nsfw.run(self.model.predictions, feed_dict={self.model.input: frame})
		result_dict.update({NSFW_SCORE_KEY: scores[0][1]})



		# print('Scores now are', scores)
		self.frame_processed_signal.emit(result_dict)

		


	def on_frame_processed(self, result_dict):
		self.buffer_channel.append(result_dict[CHANNEL_ID_KEY])
		self.buffer_channel2.append(result_dict[SECOND_CHANNEL_ID_KEY])
		self.buffer_nsfw.append(result_dict[NSFW_SCORE_KEY])
		if len(self.buffer_channel) >= self.buffer_size:
			self.nsfw_decision = "Безопасно"
			self.tv_decision     = "ТВ"

			# находим среднюю вероятность nsfw-контента
			nsfw_score = 1.0 * sum(self.buffer_nsfw) / len(self.buffer_nsfw)
			result_nsfw = nsfw_score > NSFW_THRESHOLD
			
			# считаем сколько раз определился каждый канал 
			channels_dict = {}
			for channel_id, confidence in self.buffer_channel:
				if channel_id in channels_dict:
					channels_dict.update({channel_id: (channels_dict[channel_id][0] + 1,\
							channels_dict[channel_id][1] + confidence)})
				else:
					channels_dict.update({channel_id: (1, confidence)})

			# находим канал, который попался наибольшее число раз
			max_value = 0
			result_channel_id = UNKNOWN_CHANNEL_ID
			print(channels_dict)
			for k in channels_dict:
				v = channels_dict[k]
				if v[0] > max_value:
					max_value = v[0]
					result_channel_id = k

			# если канал опредён, вычислим уровень достоверности (вероятность)
			channel_confidence = 0
			if result_channel_id != UNKNOWN_CHANNEL_ID:
				channel_confidence = channels_dict[result_channel_id][1] / channels_dict[result_channel_id][0]
				if channel_confidence > 1:
					channel_confidence = 1


			
			# считаем сколько раз определился каждый канал 
			channels_dict2 = {}
			for channel_id, confidence in self.buffer_channel2:
				if channel_id in channels_dict2:
					channels_dict2.update({channel_id: (channels_dict2[channel_id][0] + 1,\
							channels_dict2[channel_id][1] + confidence)})
				else:
					channels_dict2.update({channel_id: (1, confidence)})

			# находим канал, который попался наибольшее число раз
			max_value2 = 0
			result_channel_id2 = UNKNOWN_CHANNEL_ID
			print(channels_dict2)
			for k in channels_dict2:
				v = channels_dict2[k]
				if v[0] > max_value2:
					max_value2 = v[0]
					result_channel_id2 = k

			# если канал опредён, вычислим уровень достоверности (вероятность)
			channel_confidence2 = 0
			if result_channel_id2 != UNKNOWN_CHANNEL_ID:
				channel_confidence2 = channels_dict2[result_channel_id2][1] / channels_dict2[result_channel_id2][0]
				if channel_confidence2 > 1:
					channel_confidence2 = 1

			# Заполнение лого каналов в приложении
			"""if result_channel_id == UNKNOWN_CHANNEL_ID:
				self.logTextBox.widget.setTextColor(QColor(153,153,0))
				self.logTextBox.widget.append("Канал не распознан")
				self.logTextBox.widget.setTextColor(QColor(0,0,0))
			el"""
			if result_channel_id != self.actual_channel_id:
				self.previous_channel_id = self.actual_channel_id
				self.actual_channel_id = result_channel_id
				self.logTextBox.widget.setTextColor(QColor(255,0,0))
				self.logTextBox.widget.append("Изменился канал с {} на {}".format(self.logo_detector.get_label(self.previous_channel_id), self.logo_detector.get_label(self.actual_channel_id)))
				self.logTextBox.widget.setTextColor(QColor(0,0,0))

			if result_channel_id2 != self.actual_channel_id2:
				self.previous_channel_id2 = self.actual_channel_id2
				self.actual_channel_id2 = result_channel_id2
				self.logTextBox.widget.setTextColor(QColor(255,0,0))
				self.logTextBox.widget.append("Внимание! Задетектировано два лого каналов!")
				self.logTextBox.widget.setTextColor(QColor(0,0,0))

			if result_nsfw != self.is_porn_detected:
				if result_nsfw:
					self.logTextBox.widget.setTextColor(QColor(255,0,0))
					self.logTextBox.widget.append("Обнаружение порно контента")
					self.logTextBox.widget.setTextColor(QColor(0,0,0))
					self.is_porn_detected = True
					self.content_value_label.setText("<font color='red'>Небезопасно</font>")
					self.tv_decision = "Порно"
				else:
					self.logTextBox.widget.setTextColor(QColor(76,153,0))
					self.logTextBox.widget.append("Возвращение к нормальному контенту")
					self.logTextBox.widget.setTextColor(QColor(0,0,0))
					self.is_porn_detected = False
					self.content_value_label.setText("<font color='green'>Безопасно</font>")
					self.tv_decision = "ТВ"

			self.channel_value_label.setText('{:25}'.format(self.logo_detector.get_label(self.actual_channel_id) +\
					' ' + self.tv_decision))
			self.channel_conf_label.setText('{:>5}%'.format(int(channel_confidence * 100)))
			self.content_conf_label.setText('{:>5}%'.format(int(nsfw_score * 100)))
			
			# Запись в лог-файл logfile.txt
			with open('logfile.txt', 'a') as the_file:
				the_file.write("time: {} | channel: {} | channel_confidence: {}% | content_confidence: {} %\n".\
						format(str(time.ctime(time.time())), self.logo_detector.get_label(self.actual_channel_id),\
						int(channel_confidence * 100), str(nsfw_score*100)[:5]))

			self.buffer_nsfw = []
			self.buffer_channel = []
			






	def nextFrameSlot(self):
		self.count_frame += 1
		ret, frame = self.cap.read()
		if ret:
			interval = self.frames_per_detection
			if self.actual_channel_id == INITIAL_CHANNEL_ID or self.actual_channel_id == UNKNOWN_CHANNEL_ID:
				interval = int(interval / 3)
			if self.count_frame % interval == 0:
				thread = Thread(target=self.process_frame, args=[frame])
				thread.start()
			if self.gc_thread == None:
				self.gc_thread =Thread(target=gc.collect, args = [])
				self.gc_thread.start()

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
			pix = QPixmap.fromImage(img)
			pix = pix.scaled(800, 450)
			self.video_frame.setPixmap(pix)
		else:
			self.cap = cv2.VideoCapture(str(self.filename))

		#gc.collect()

	def start(self):
		self.frame_processed_signal.connect(self.on_frame_processed)
		self.timer.timeout.connect(self.nextFrameSlot)
		#self.timer.timeout.connect(self.get_scores)
		self.timer.start(1000.0 / self.frame_rate)

	def deleteLater(self):
		self.cap.release()
		self.logo_detector.close()
		super(QWidget, self).deleteLater()


class ControlWindow(QMainWindow):
	def __init__(self, videoFileName, frames_per_detection, buffer_size):
		super(ControlWindow, self).__init__()
		self.setGeometry(5, 5, 1200, 400)
		self.setWindowTitle("Детекция контента и логотипа")

		self.capture_window = None

		self.videoDisplayWidget = VideoDisplayWidget(self)
		self.setCentralWidget(self.videoDisplayWidget)
		self.capture_window = VideoCapture(videoFileName, int(frames_per_detection), int(buffer_size),
										   self.videoDisplayWidget)
		self.capture_window.start()


class VideoDisplayWidget(QWidget):
	def __init__(self, parent):
		super(VideoDisplayWidget, self).__init__(parent)
		#self.layout = QFormLayout(self)
		#self.setLayout(self.layout)
		self.layout = QGridLayout(self)
		self.setLayout(self.layout)

if __name__ == '__main__':
	app = QApplication(sys.argv)

	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input_video", default="udp://localhost:1235", help="Path to the input video")

	parser.add_argument("-f", "--frames_per_detection", default=7, help="Frames per detection")

	parser.add_argument("-b", "--buffer_size", default=11, help="Buffer size")

	args = parser.parse_args()

	window = ControlWindow(args.input_video, args.frames_per_detection, args.buffer_size)
	window.show()
	sys.exit(app.exec_())
	#os.system("ps axf | grep vlc | grep -v grep | awk '{print \"kill -9 \" $1}' | sh")
