import os
import yaml

class PipelineConfig(object):
    DETECTION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '../models/CRNN_v4.0.1')

    RECONGNITION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '../models/N_EAST_v0.0.0')

    # CHINESE_RECONGNITION_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                                     '../models/CRNN_tf_v9_7_236k')

    # CUT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                                     '../models/Text_seg_v1.0.1')