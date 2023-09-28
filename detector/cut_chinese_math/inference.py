from detector.cut_chinese_math.nets.net_factory import get_net
import tensorflow as tf
import numpy as np
import cv2
from detector.cut_chinese_math.cut_config import Config as config
import os
from glob import glob
from tqdm import tqdm
from detector.cut_chinese_math.post_process import get_classify,enhance_precision,prevent_cut
import time

class Inference():
    def __init__(self,sess,model_path):
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        model = get_net(config.MODEL_TYPE)
        model.build_net(self.input_images, self.is_training)

        logits = model.logits

        self.logits = tf.argmax(logits, axis=-1)

        heat_map = model.heat_map

        heat_map = heat_map[..., tf.newaxis]

        max_pooled_in_tensor = tf.nn.pool(heat_map, window_shape=[4, 1], pooling_type='MAX', padding='SAME')

        tensor_peaks = tf.where(tf.equal(heat_map, max_pooled_in_tensor), heat_map,
                                tf.zeros_like(heat_map))

        tensor_peaks = tensor_peaks[:, :, 0, 0]

        self.points = tf.where(tensor_peaks > config.DELETE_LINE_THRESHOLD)

        self.heat_map = heat_map[:, :, 0, 0]

        saver = tf.train.Saver()

        self.sess = sess

        saver.restore(self.sess,model_path)

    def sigle_inference_resize(self,img,display = False,save_path = ''):
        h,w,_ = img.shape
        img_temp = cv2.resize(img, (config.IMG_W, config.IMG_H))
        img_temp = img_temp[np.newaxis, ...]

        logits_, heat_map_, points, = self.sess.run([self.logits, self.heat_map, self.points],
                                                      feed_dict={self.input_images: img_temp, self.is_training: False})

        heat_map_ = heat_map_[0, :]
        logits_ = logits_[0, :]

        #get_points
        points_dict = {}
        for point in points:
            if point[0] not in points_dict:
                points_dict[point[0]] = [point[1]]
            else:
                points_dict[point[0]].append(point[1])
        if points_dict:
            points_list = points_dict[0]
            points_list.sort()
        else:
            points_list = []

        categories_list = self.get_classify(logits_,points_list)

        logits_ = logits_[np.newaxis,:,np.newaxis]
        logits_ = cv2.resize(logits_,(w,1),interpolation=cv2.INTER_NEAREST)
        logits_ = logits_[0,:]

        points_list = list(map(lambda x: int(x * w / config.HEAT_MAP_SIZE), points_list))


        if display:
            img = display_result(img,points_list,categories_list,save_path)
            return points_list, categories_list, img

        return points_list,categories_list,logits_

        # self.display_result(img,points_list,categories_list)

    def sigle_inference_padding(self,img,display = False,save_path = ''):

        h,w,_ = img.shape
        w_resize = config.IMG_H/h

        img_temp = cv2.resize(img, (int(w*w_resize), config.IMG_H))
        img_temp = img_temp[np.newaxis, ...]

        logits_, heat_map_, points, = self.sess.run([self.logits, self.heat_map, self.points],
                                                    feed_dict={self.input_images: img_temp, self.is_training: False})
        heat_map_ = heat_map_[0, :]
        logits_ = logits_[0, :]

        cond_list = []

        # get_points
        points_dict = {}
        for point in points:
            cond_list.append(heat_map_[point[1]])
            if point[0] not in points_dict:
                points_dict[point[0]] = [point[1]]
            else:
                points_dict[point[0]].append(point[1])
        if points_dict:
            points_list = points_dict[0]
            points_list.sort()
        else:
            points_list = []

        categories_list = get_classify(logits_, points_list)



        logits_ = logits_[np.newaxis, :, np.newaxis]
        logits_ = cv2.resize(logits_, (w, 1), interpolation=cv2.INTER_NEAREST)
        logits_ = logits_[0, :]

        heat_map_ = heat_map_[np.newaxis, :, np.newaxis]
        heat_map_ = cv2.resize(heat_map_, (w, 1), interpolation=cv2.INTER_NEAREST)
        heat_map_ = heat_map_[0, :]


        points_list = list(map(lambda x: int(x*2 /w_resize), points_list))

        if config.POST_PROCESS:

            points_list, categories_list, cond_list = prevent_cut(img,points_list,logits_,heat_map_,categories_list)

            points_list, categories_list, cond_list = enhance_precision(points_list, categories_list, logits_, cond_list)



        if display:
            display_result(img,points_list,categories_list,save_path)

        return points_list,cond_list, categories_list, logits_

    def batch_inference_padding(self, img_list,display=False, save_path=''):
        input_batch,w_resize_list,w_list = self.create__batch_input(img_list)

        logits_, heat_map_, points, = self.sess.run([self.logits, self.heat_map, self.points],
                                                    feed_dict={self.input_images: input_batch, self.is_training: False})

        all_points_list = []
        all_cond_list = []
        all_categories_list = []
        all_logits = []

        for i in range(input_batch.shape[0]):
            _,img_w,_ = img_list[i].shape
            img_w = int(img_w/2)
            heat_map_temp = heat_map_[i, :img_w]
            logits_temp = logits_[i, :img_w]
            cond_list = []
            w = w_list[i]
            w_resize = w_resize_list[i]

            # get_points
            points_dict = {}
            for point in points:
                if point[0] == i and point[1] < img_w:
                    cond_list.append(heat_map_temp[point[1]])
                    if point[0] not in points_dict:
                        points_dict[point[0]] = [point[1]]
                    else:
                        points_dict[point[0]].append(point[1])
            if points_dict:
                points_list = points_dict[i]
                points_list.sort()
            else:
                points_list = []

            categories_list = get_classify(logits_temp, points_list)

            # logits_temp = logits_temp[np.newaxis, :, np.newaxis]
            logits_temp = cv2.resize(logits_temp, (w, 1), interpolation=cv2.INTER_NEAREST)
            logits_temp = logits_temp[0, :]


            # heat_map_temp = heat_map_temp[np.newaxis, :, np.newaxis]

            heat_map_temp = cv2.resize(heat_map_, (w, 1), interpolation=cv2.INTER_NEAREST)


            heat_map_temp = heat_map_temp[0, :]

            points_list = list(map(lambda x: int(x * 2 / w_resize), points_list))

            if config.POST_PROCESS:
                points_list, categories_list, cond_list = prevent_cut(img_list[i], points_list, logits_temp, heat_map_temp, categories_list)

                points_list, categories_list, cond_list = enhance_precision(points_list, categories_list, logits_temp,
                                                                            cond_list)

            if display:
                display_result(img, points_list, categories_list, save_path)

            all_points_list.append(points_list)
            all_cond_list.append(cond_list)
            all_categories_list.append(categories_list)
            all_logits.append(logits_temp)

        end = time.time()


        return all_points_list, all_cond_list, all_categories_list, all_logits


    def create__batch_input(self,img_list):
        max_w = 0
        img_list_temp = []
        w_resize_list = []
        w_list = []
        for img in img_list:
            h, w, _ = img.shape
            w_resize = config.IMG_H / h

            img_temp = cv2.resize(img, (int(w * w_resize), config.IMG_H))
            w_resize_list.append(w_resize)
            w_list.append(w)
            _,w,_ = img_temp.shape
            if w > max_w:
                max_w = w
            img_temp = img_temp[np.newaxis,...]
            img_list_temp.append(img_temp)

        input_batch = np.zeros([len(img_list_temp),config.IMG_H,max_w,3],dtype=np.float32)
        for i,img in enumerate(img_list_temp):
            _,h,w,_ = img.shape
            # img = img[np.newaxis, ...]
            input_batch[i,:,:w,:] = img

        return input_batch,w_resize_list,w_list



    def get_classify(self,logits,points_list):
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

                categories = np.mean(logits_list == 0)

                if categories > 0.5:
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

def display_result(img,point_list,categories_list,save_path):
    img_temp = np.zeros(img.shape,dtype=np.uint8)
    _,w,_ = img.shape

    if point_list:
        for num in range(len(point_list)+1):
            if num == 0:
                begin = 0
                end = point_list[num]
                img[:, end, :] = (0, 255, 0)
            elif num == len(point_list):
                begin = point_list[num-1]
                end = w
            else:
                begin = point_list[num-1]+1
                end = point_list[num]
                img[:, end, :] = (0, 255, 0)

            if categories_list[num] == 0:
                rgb = (255,0,0)
            else:
                rgb = (0,0,255)

            img_temp[:,begin:end,:] = rgb


    else:
        categories = categories_list[0]
        if categories == 0:
            rgb = (255, 0, 0)
        else:
            rgb = (0, 0, 255)

        img_temp[:, :, :] = rgb


    img = cv2.addWeighted(img,0.7,img_temp,0.3,0)

    # cv2.imwrite(save_path,img)

    return img












if __name__ == '__main__':
    img_path = '/home/wzh/NB/DATA/切分/img2/2B92012A59E944D6A32A899E9AD11AFD_5.jpg'
    img_floder = '/home/wzh/NB/DATA/点阵数据/test/val_cut'
    inference = Inference(config.PRETRAINED_MODEL_PATH)
    save_path = '/home/wzh/NB/DATA/点阵数据/test/val_cut_display'
    for img_path in tqdm(glob(os.path.join(img_floder,'*.png'))):
        img = cv2.imread(img_path)
        img_name = img_path.split('/')[-1]
        save_path_temp = os.path.join(save_path,img_name)

        inference.sigle_inference_resize(img,True,save_path_temp)



