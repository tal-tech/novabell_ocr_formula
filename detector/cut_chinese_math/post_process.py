import numpy as np
import cv2
from detector.cut_chinese_math.cut_config import Config as config


def get_classify(logits,points_list):
    categories_list = []
    if points_list:
        for num in range(len(points_list)+1):

            if num == 0:
                begin = 0
                end = points_list[num]
            elif num == len(points_list):
                begin = points_list[num-1]
                end = len(logits)
            else:
                begin = points_list[num-1]+1
                end = points_list[num]

            logits_list = logits[begin:end]

            categories_num = np.mean(logits_list == 0)
            categories_chinese = np.mean(logits_list == 1)

            if categories_num > categories_chinese:
                categories = 0
            else:
                categories = 1

            categories_list.append(categories)
    else:
        categories = np.mean(logits == 0)

        if categories > 0.5:
            categories = 0
        else:
            categories = 1

        categories_list.append(categories)

    return categories_list


def enhance_precision(points_list, categories_list, logits,cond_list):

    num = 0

    while num+1 <= len(categories_list)-1:
        if categories_list[num] == categories_list[num+1]:
            del points_list[num]
            del cond_list[num]
            categories_list = get_classify(logits,points_list)
        else:
            num = num+1

    return points_list,categories_list,cond_list


def prevent_cut(img,points_list,logits,heat_map,categories_list):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w = img.shape
    img = np.asarray(img<250,np.int32)
    h_shadow = np.sum(img,0)
    points_list_temp = []
    cond_list_temp = []
    for num,point in enumerate(points_list):
        if h_shadow[point] > config.MAX_CUT_POINT:
            for i in range(max(5,int(w/config.PREVENT_CUT_MOVE))):
                if h_shadow[max(point-i,0)] <=config.MAX_CUT_POINT:
                    points_list_temp.append(point-i)
                    cond_list_temp.append(heat_map[point-i])
                    break
                elif h_shadow[min(point+i,w-1)] <= config.MAX_CUT_POINT:
                    points_list_temp.append(point + i)
                    cond_list_temp.append(heat_map[point + i])
                    break

        else:
            points_list_temp.append(point)
            cond_list_temp.append(heat_map[point])

    if points_list_temp!= points_list:
        points_list_temp = list(set(points_list_temp))
        points_list_temp.sort()
        categories_list = get_classify(logits, points_list_temp)

    return points_list_temp,categories_list,cond_list_temp






