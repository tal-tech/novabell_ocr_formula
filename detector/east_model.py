import os

import time
import datetime
import cv2
import numpy as np


import functools
import logging
import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('loading model')
import tensorflow as tf
import detector.model as model

from detector.eval import resize_image, sort_poly, detect
import natsort
from detector.cut_chinese_math.inference import Inference as CutModel
from detector.detector_config import Config as config
from detector.locality_aware_nms import big_w_nms

@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret



class ModelEast(object):
    def __init__(self,session=None, graph=None, de_model_dir='', gpu_opt='GPU'):
        if gpu_opt == "CPU":
            self._device = "/cpu:0"
        else:
            self._device = "/gpu:0"

        if not session:
            self.sess = tf.Session(config=self._get_sess_config())
        else:
            self.sess = session
        de_meta_file, de_ckpt_path = self._get_model_path(de_model_dir)

        with graph.as_default():
            with tf.device(self._device):

                self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

                self.f_score, self.f_geometry = model.model(self.input_images, is_training=False)

                variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
                self.saver = tf.train.Saver(variable_averages.variables_to_restore())

                logger.info('Restore from {}'.format(de_ckpt_path))
                self.saver.restore(self.sess, de_ckpt_path)


    def restore(self,model_path):
        self.saver.restore(self.sess,model_path)


    def _get_model_path(self, model_dir):
        files = natsort.natsorted(os.listdir(model_dir))
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        # elif len(meta_files)>1:
        # raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = os.path.join(model_dir, meta_files[-1])

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            raise RuntimeError('No checkpoint file found')

        return meta_file, ckpt_path


    def inference(self, img):
        img_temp = img.copy()
        h, w, _ = img_temp.shape
        det_timer = {}
        time_0 = time.time()
        if len(config.MUTIL_SIZE) > 0:
            boxes = []
            for img_size in config.MUTIL_SIZE:
                boxes.extend(self.network_inference(img, img_size)['text_lines'])

            box_list = []
            for box in boxes:
                box = [box['x0'], box['y0'], box['x1'], box['y1'], box['x2'], box['y2'], box['x3'], box['y3'],
                       box['score']]
                box_list.append(box)

            text_lines = []

            if box_list:
                boxes = np.asarray(box_list, np.float32)
                # boxes = nms_locality.standard_nms(boxes, 0.1)
                boxes = big_w_nms(boxes, 0.1)
                scores = boxes[:, 8].reshape(-1)
                boxes = boxes[:, :8].reshape((-1, 4, 2))

                if boxes.any():
                    text_lines = []
                    for box, score in zip(boxes, scores):
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue
                        tl = collections.OrderedDict(zip(
                            ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                            map(float, box.flatten())))
                        tl['score'] = float(score)
                        text_lines.append(tl)
            ret = {'text_lines': text_lines}
        else:
            ret = self.network_inference(img, config.IMAGE_MAX_SIZE)
        time_1 = time.time()
        det_timer["det_model"] = time_1 - time_0
        ret.update(get_host_info())

        box_list = self.normalizate_output(ret)

        return box_list, det_timer 


    def network_inference(self,img,img_size):
        im_resized, (ratio_h, ratio_w) = resize_image(img,img_size)
        score, geometry = self.sess.run(
            [self.f_score, self.f_geometry],
            feed_dict={self.input_images: [im_resized[:, :, ::-1]]})
        boxes = detect(score_map=score, geo_map=geometry)
        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {'text_lines': text_lines}

        return ret


    def normalizate_output(self,ret):
        boxes_temp = ret['text_lines']
        boxes = []
        for box_temp in boxes_temp:
            x1 = max(min(box_temp['x0'],box_temp['x1'],box_temp['x2'],box_temp['x3']),0)
            x2 = max(box_temp['x0'],box_temp['x1'],box_temp['x2'],box_temp['x3'])
            y1 = max(min(box_temp['y0'],box_temp['y1'],box_temp['y2'],box_temp['y3']),0)
            y2 = max(box_temp['y0'],box_temp['y1'],box_temp['y2'],box_temp['y3'])
            boxes.append({'label':0,'parent':[int(x1),int(y1),int(x2),int(y2)]})
        return boxes


def display_reuslt(result_list,img):

    img, x_pro, y_pro = img_resize(img)
    color = (0,255,0)
    for result in result_list:
        bbox = result
        cv2.rectangle(img, (int(bbox[0] * x_pro), int(bbox[1] * y_pro)),
                      (int(bbox[2] * x_pro), int(bbox[3] * y_pro)), color, 4)

    return img


def img_resize(img):
    '''
    讲图像改变为统一大小，并返回缩放比例
    :param img:
    :return:
    '''
    x_pro = 3024 / img.shape[1]
    y_pro = 4031 / img.shape[0]
    img = cv2.resize(img, (3024, 4032))
    return img,x_pro,y_pro


def get_detector_sess():
    tmp_graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config,graph=tmp_graph)
    return sess,tmp_graph


if __name__ == '__main__':
    checkpoint_path = '/home/wzh/NB/model/open_source_east/'
    img_path = '/home/wzh/NB/DATA/NB_DATA/sea/all_data/11.jpg'
    # img_path = '/home/wzh/NB/DATA/点阵数据/2019-05-14-12-27-29.jpeg'
    img_name = img_path.split('/')[-1]
    sess, tmp_graph = get_detector_sess()
    model_east = ModelEast(sess, tmp_graph,checkpoint_path)
    img = cv2.imread(img_path)
    ret = model_east.inference(img)
    # ret.sort(key=lambda x:x[2]-x[0])
    img = display_reuslt(ret,img)
    cv2.imwrite(img_name,img)
