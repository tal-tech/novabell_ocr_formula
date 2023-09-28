from shapely.geometry import Polygon
import numpy as np
def area(point1,point2):
    return max((point2[0]-point1[0]),0)*max((point2[1]-point1[1]),0)


# def get_iou_small(box1,box2):
#     '''
#     box1在左边，box在右边
#     :param box1:
#     :param box2:
#     :return:
#     '''
#     left = Polygon(box1[:8].reshape((4, 2)))
#     right = Polygon(box2[:8].reshape((4, 2)))
#     if not left.is_valid or not right.is_valid:
#         return 0
#     inter = left.intersection(right).area
#
#     if left.area > right.area:
#         small = 2
#     else:
#         small = 1
#
#     return inter/min(left.area,right.area),small

def quadrangle_to_bbox(bbox):
    if len(bbox) == 4:
        return bbox
    else:
        top = min(bbox[1], bbox[3], bbox[5], bbox[7])
        bottom = max(bbox[1], bbox[3], bbox[5], bbox[7])
        left = min(bbox[0], bbox[6], bbox[2], bbox[4])
        right = max(bbox[0], bbox[6], bbox[2], bbox[4])
        bbox = [left, top, right, bottom]
        return bbox

def get_iou_small(box1,box2):
    box1 = quadrangle_to_bbox(box1[:8])
    box2 = quadrangle_to_bbox(box2[:8])
    area1 = area((box1[0],box1[1]),(box1[2],box1[3]))
    area2 = area((box2[0],box2[1]),(box2[2],box2[3]))
    if area1 == 0 or area2 == 0:
        return 0

    point1 = (max(box1[0],box2[0]),max(box1[1],box2[1]))
    point2 = (min(box1[2],box2[2]),min(box1[3],box2[3]))

    area3 = area(point1,point2)
    if area1 >area2:
        num = 2
    else:
        num = 1

    return area3/min(area1,area2),num

def h_iou(box1,box2,type='noemal'):
    '''
      box1在左边，box在右边
      :param box1:
      :param box2:
      :return:
      '''
    max_top = max(box1[3],box2[1])
    min_bottom = min(box1[5],box2[7])
    left_h = box1[5]-box1[3]
    right_h = box2[7] - box2[1]

    if max_top>=min_bottom:
        return 0
    elif type == 'min':
        return (min_bottom-max_top)/min(left_h,right_h)
    elif type == 'normal':
        return (min_bottom-max_top)/left_h+right_h-(min_bottom-max_top)

def w_iou(column1,column2,type = 'min'):
    max_left = max(column1[0],column2[0])
    min_right = min(column1[2],column2[2])

    if max_left>=min_right:
        return 0
    else:               #分为与大的算iou,小的算iou,普通的iou算法
        if type =='min':
            return (min_right-max_left)/min(column1[2]-column1[0],column2[2]-column2[0])
        elif type == 'max':
            return (min_right-max_left)/max(column1[2]-column1[0],column2[2]-column2[0])
        elif type == 'normal':
            return (min_right - max_left) / (column1[2] - column1[0])+( column2[2] - column2[0])-(min_right - max_left)



def delete_overlap(boxes):
    delete_list = []
    for i in range(boxes.shape[0]):
        for j in range(i+1,boxes.shape[0]):
            iou_min, num = get_iou_small(boxes[i], boxes[j])
            if iou_min>0.9:
                if num == 1:
                    delete_list.append(i)
                elif num ==2:
                    delete_list.append(j)
    return np.delete(boxes,delete_list,0)

# def merge_box(boxes):
#     merge_list = []
#     for i in range(boxes.shape[0]):
#         for j in range(i + 1, boxes.shape[0]):
#             iou_min, num = get_iou_small(boxes[i], boxes[j])
#             if iou_min > 0.9:
#                 if num == 1:
#                     delete_list.append(i)
#                 elif num == 2:
#                     delete_list.append(j)
#     return np.delete(boxes, delete_list, 0)

def reprocessing(boxes):
    boxes = delete_overlap(boxes)
    return boxes