
class Config(object):
    WEIGHTED_MERGE_TYPE = 'w_max'  # 'normal' / 'w_max'

    REPROCESS = False

    SCORE_MAP_THRESH = 0.8

    BOX_THRESH = 0.1

    NMS_THRES = 0.2

    FILTER_BOX_TWICE = True

    SCORE_MAP_ZOOM = 0.3

    IMAGE_MAX_SIZE = 768

    MUTIL_SIZE = [] # []/[384,512,768]

    USE_CUT = False


