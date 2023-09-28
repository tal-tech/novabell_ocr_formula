# code: utf-8

import os
import sys
import time

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import math
import io
import cv2
import natsort

import recognizor.hqr_utils as hqr_utils
from recognizor.recognizor_base import RecognizorBase
from recognizor.hqr_utils import CONFIG


class FormulaRecognizor(RecognizorBase):


    def __init__(self,
                model_dir=None,
                session=None,
                graph=None,
                gpu_opt='GPU',
                decode_type='greedy'):

        super().__init__(model_dir,
                        session,
                        graph,
                        gpu_opt)
        self.charset = hqr_utils.get_CharSet_list()
        self.charset.append("")
        # Build inference graph.
        with self._graph.as_default():
            self._inputs = self._model_graph.get_tensor_by_name("inputs:0")
            self._width = self._model_graph.get_tensor_by_name("width:0")
            self._training_mode = self._model_graph.get_tensor_by_name("training_mode:0")
            # control inference dropout.
            self._dropout_in_keep = self._model_graph.get_tensor_by_name("dropout_in_keep:0")
            self._dropout_out_keep = self._model_graph.get_tensor_by_name("dropout_out_keep:0")
            self._seq_len = self._model_graph.get_tensor_by_name("tower_0/cnn/Mul:0")

            self.logits=self._model_graph.get_tensor_by_name("logits:0")

            if decode_type == 'beam':
                self.output,self.decode_prob = self._build_beam_search_decoder(self.logits,self._seq_len)
            else:
                self.output,self.decode_prob = self._build_greedy_search_decoder(self.logits,self._seq_len)


    def inference(self,image_list):
        result = self._predict_batch(image_list)
        return result


    def decode_hyp(self,hypothesis):
        decode_result = "".join([self.charset[hypothesis[c]] for c in range(hypothesis.shape[0])])
        return decode_result

    
    def _preprocess(self, image, img_height=CONFIG["IMG_HEIGHT"]):
        tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tmp_h, tmp_w = tmp_image.shape[:2]

        tmp_ratio = img_height / tmp_h
        target_w = max(64,int(tmp_ratio * tmp_w))

        tmp_image = cv2.resize(tmp_image, (target_w, img_height))

        if CONFIG['NORMALIZATION']:
            tmp_image = tmp_image / 255 * 2 - 1

        width = tmp_image.shape[1]

        return tmp_image,width


    def _predict_single(self,image):

        return self._predict_batch(image_list=[image],batch_size=1)

    
    def _predict_batch(self,image_list,batch_size=32):
        """_predict_batch [summary]
        
        Args:
            image_list ([type]): [description]
            batch_size (int, optional): [description]. Defaults to 32.
        
        Returns:
            result_batch: the recognization results corresponding to the image_list.
            timer: the process log of recognizor.
        """
        image_batch = []
        width_batch = []
        result_batch = []
        timer = {"preprocess" : 0.0,
                "rec_model" : 0.0,
                "decode" : 0.0,
                "batches":0}

        run_batch = int(math.ceil(1.0*len(image_list) / batch_size))
        timer["batches"] = run_batch
        timer["num_lines"] = len(image_list)
        # run_batch = (len(image_list) // batch_size) + 1
        for i in range(run_batch):
            time_0 = time.time()
            start_idx = i * batch_size
            end_idx = min((i+1)*batch_size,len(image_list))

            if start_idx == end_idx:
                break

            image_batch = []
            width_batch = []
            tmp_max_width = 0
            for image in image_list[start_idx:end_idx]:
                tmp_image, width = self._preprocess(image)
                image_batch.append(tmp_image)
                width_batch.append(width)
                if width > tmp_max_width:
                    tmp_max_width = width
            image_batch_np = np.zeros([end_idx-start_idx,CONFIG["IMG_HEIGHT"],tmp_max_width])
            for image_idx,image in enumerate(image_batch):
                image_batch_np[image_idx,:,0:image.shape[1]] = image
            time_1 = time.time()
            timer["preprocess"] += (time_1 - time_0)
            
            tmp_feed_dict = {}
            tmp_feed_dict[self._inputs] = image_batch_np[..., np.newaxis]
            tmp_feed_dict[self._width] = np.array(width_batch)
            tmp_feed_dict[self._training_mode] = False
            tmp_feed_dict[self._dropout_in_keep] = 1.0
            tmp_feed_dict[self._dropout_out_keep] = 1.0

            tmp_hyps = self._sess.run(self.output,feed_dict=tmp_feed_dict)
            time_2 = time.time()
            timer["rec_model"] += (time_2 - time_1)
            for tmp_idx in range(tmp_hyps.shape[0]):
                tmp_decode_result = self.decode_hyp(tmp_hyps[tmp_idx])
                result_batch.append(tmp_decode_result)
            time_3 = time.time()
            timer["decode"] += (time_3 - time_2)

        return result_batch,timer


class TextRecognizor(object):

    
    def __init__(self,
                model_sess,
                model_graph,
                model_dir):
        # model config

        # init recognize model.
        # config parameters (model_dir, tf_session, tf_graph)
        fr_model_dir = model_dir
        fr_sess = model_sess
        fr_graph = model_graph
        self.FR = FormulaRecognizor(fr_model_dir, fr_sess, fr_graph)


    def inference(self,image,det_result):
        """inference [summary]
        
        Args:
            image ([type]): [description]
            det_result ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        recognizor_result = []
        tmp_image_list = []
        time_0 = time.time()
        for line_idx, det_line in enumerate(det_result):
            cut_line = TextRecognizor.get_det_img(image,det_line)
            tmp_image_list.append(cut_line)
        time_1 = time.time()

        recognizor_result, recognizor_timer = self.FR.inference(tmp_image_list)
        recognizor_timer["cut_images"] = (time_1 - time_0) 
        return recognizor_result, recognizor_timer


    @staticmethod
    def get_det_img(img, single_det_result):
        box = single_det_result['parent']
        h, w, _ = img.shape
        crop_img = img[max(box[1], 0):min(box[3] + 1, h), max(box[0], 0):min(box[2] + 1, w), :]
        return crop_img