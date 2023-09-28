class Config(object):
    NUM_CLASS = 3

    RNN_UNITS = 256

    SIGMAN = 4.0

    IMG_H = 64

    IMG_MAX_W = IMG_H*10

    CLASSIFY_MAP_SIZE = IMG_H*5

    HEAT_MAP_SIZE = IMG_H*5

    GPUS = [0]

    MOVING_AVERAGE_DECAY =0.997

    MODEL_TYPE = 'cnn_64'

    TYPE_DICT = {'非中文字符\n':0,'中文字符\n':1,'非中文字符':0,'中文字符':1,'中文\n':1,'非中文\n':0}

    THRESHOLD = 20

    DELETE_LINE_THRESHOLD = 0.3

    MAX_CUT_POINT = 0  #允许切割线切到几个像素点的笔画

    PREVENT_CUT_MOVE = 20 #允许左右移动多少像素

    POST_PROCESS = True
