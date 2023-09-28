import numpy as np
from shapely.geometry import Polygon
from detector.detector_config import Config as config


def intersection(g, p):
    #计算iou
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g

def weighted_merge_max(g, p):
    g[0] = min(g[0],p[0])
    g[1] = min(g[1],p[1])
    g[2] = max(g[2],p[2])
    g[3] = min(g[3],p[3])
    g[4] = max(g[4],g[4])
    g[5] = max(g[5],g[5])
    g[6] = min(g[6],g[6])
    g[7] = max(g[7],g[7])
    g[8] = (g[8] + p[8])
    return g

def weighted_merge_w_max(g, p):
    g[0] = min(g[0],p[0])
    # g[1] = min(g[1],p[1])
    g[1] = (g[1] * g[8] + p[1] * p[8])/(g[8] + p[8])
    g[2] = max(g[2],p[2])
    # g[3] = min(g[3],p[3])
    g[3] = (g[3] * g[8] + p[3] * p[8]) / (g[8] + p[8])
    g[4] = max(g[4],g[4])
    # g[5] = max(g[5],g[5])
    g[5] = (g[5] * g[8] + p[5] * p[8]) / (g[8] + p[8])
    g[6] = min(g[6],g[6])
    # g[7] = max(g[7],g[7])
    g[7] = (g[7] * g[8] + p[7] * p[8]) / (g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]

def big_w_nms(S,thres):
    w = np.max(S[:,[1,7]],-1)-np.min(S[:,[0,6]],-1)
    order = np.argsort(w)[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]

def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            if config.WEIGHTED_MERGE_TYPE == 'normal':
                p = weighted_merge(g, p)
            elif config.WEIGHTED_MERGE_TYPE == 'w_max':
                p = weighted_merge_w_max(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return S
    # return standard_nms(np.array(S), thres)


if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
