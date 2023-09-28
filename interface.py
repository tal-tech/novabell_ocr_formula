# code:utf-8
import os
import sys
import cv2
import json
import functools
import numpy as np
import tensorflow as tf
import time
cv2.useOptimized()

sys.path.append("./Recognizor")
from recognizor.text_recognizor import TextRecognizor
sys.path.append("./Detector")
from detector.east_model import ModelEast
from pipe_utils.pipeline_config import PipelineConfig as config

tf.app.flags.DEFINE_string('b', '', 'kernel')
tf.app.flags.DEFINE_string('w', '', 'kernel')


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


class TextInterpreter(object):


    def __init__(self):
        det_sess, det_graph = self.get_detector_sess()
        self.detector = ModelEast(det_sess, det_graph, config.DETECTION_MODEL_PATH)

        FR_sess, FR_graph = self.get_recognizor_sess()
        self.text_recognizor = TextRecognizor(FR_sess, FR_graph, config.RECONGNITION_MODEL_PATH)


    def get_detector_sess(self):
        tmp_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=tmp_graph)
        return sess, tmp_graph


    def get_recognizor_sess(self):
        tmp_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=tmp_graph)
        return sess, tmp_graph


    def interpret(self, image):
        """interpret [summary]
        
        Args:
            image ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        # timer
        interpretor_timer = {}
        time_1 = time.time()

        # detector module
        detector_result,detector_timer = self.detector.inference(image)
        time_2 = time.time()
        interpretor_timer["detector_cost"] = (time_2-time_1)
        interpretor_timer.update(detector_timer)

        # recognizor module
        recognizor_result,recognizor_timer = self.text_recognizor.inference(image, detector_result)
        time_3 = time.time()
        interpretor_timer["recognizor_cost"] = (time_3-time_2)
        interpretor_timer.update(recognizor_timer)

        # normalise output
        final_result = self._decorate_result(detector_result,recognizor_result)
        time_4 = time.time()
        interpretor_timer["decorate_cost"] = (time_4-time_3)
        interpretor_timer["total_cost"] = (time_4-time_1)
        # interpretor_timer.update(get_host_info())
        return final_result, interpretor_timer


    def _decorate_result(self, detection_result, recognition_result):
        """_decorate_result 
            format the output results:

            "location" : the location of text box. Format is [x1,y1,x2,y2] - (x1,y1) locates 
                        the top-left point and (x2,y2) locates the bottom-right point.
            "content" : the recognize content of the text.
            "probability" : removed in this version.
            "char_count" : the char count of the whole image.

        
        Args:
            detection_result ([list]): [description]
            recognition_result ([list]): [description]
        """
        result = {}
        text = []
        char_count = 0
        for tmp_box, tmp_text in zip(detection_result, recognition_result):
            tmp_d = {}
            tmp_d["location"] = tmp_box["parent"]
            tmp_d["content"] = "".join(tmp_text)
            # tmp_d["probability"] = str(tmp_text[1])
            char_count += len(tmp_text)
            text.append(tmp_d)

        # sort by location.
        text = self._sort_by_location(text)

        result["text"] = text
        result["char_count"] = char_count

        return result


    def _sort_by_location(self, result_list):
        new_result_list = sorted(result_list,key=lambda k : ((k["location"][0]+k["location"][2])//2,
                                                             (k["location"][1]+k["location"][3])//2),reverse=False)
        return new_result_list