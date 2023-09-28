# code: utf-8

import os

import tensorflow as tf
import numpy as np
import natsort
import math
import numba as nb

## np.sum
# @nb.jit(nopython=True,parallel=True)
# def nb_expsum(x):
#     val=0
#     # for ix in nb.prange(x.shape[0]):
#         # val+=math.exp(x[ix])
#     for item in x:
#         val+=math.exp(item)
#     return val
"""
@nb.jit(nopython=True,parallel=True)
def nb_exp(x,maxValue):
    val=0
    # for ix in nb.prange(x.shape[0]):
        # val+=math.exp(x[ix])
    for item in x:
        val+=math.exp(item-maxValue)
    return 1/val
"""
## argmax 
# @nb.jit(nopython=True)
# def nb_argmax(x):

#     maxIndex = 0
#     for i in range(x.shape[0]):
#         if x[i] > x[maxIndex]:
#             maxIndex = i
#     return maxIndex
"""
@nb.jit(nopython=True,parallel=True)
def nb_softmax(x,shape):
    val=0
    maxIndex = 0
    for i in range(shape):
        val+=math.exp(x[i])
        if x[i] > x[maxIndex]:
            maxIndex = i
    softmax = math.exp(x[maxIndex])/val
    return maxIndex,softmax
"""
"""
@nb.jit(nopython=True, parallel=True)
def nb_softmax_test(x, shape):
    val = 0
    maxIndex = 0
    max = 0
    for i in range(shape):
        x_new = 1.0 +x[i]/2048.0
        for j in range(11):
            x_new = x_new * x_new
        val += x_new
        if x[i] > x[maxIndex]:
            maxIndex = i
            max = x_new
    softmax = max / val
    return maxIndex, softmax
"""

"""
@nb.jit(nopython=True,parallel=True)
def nb_sum(x):
    val=0
    for ix in nb.prange(x.shape[0]):
        val+=x[ix]
    return val
"""
#@nb.jit("float32(float32[:])", cache=False, nopython=True, nogil=True, parallel=True)
#def esum(z):
#    return np.sum(np.exp(z))


#@nb.jit(nopython=True,parallel=True)
#def totalSoftmax(x,index):
#    val=0
#    for i in range(x.shape[0]):
#        val+=math.exp(x[i])
#    temp = math.exp(x[index])/val
#    return temp

class RecognizorBase(object):
    """RecognizorBase: A tensorflow model inference base class.

    Implement basic methods & define several normal interfaces.
    """


    def __init__(self,
                model_dir=None,
                session=None,
                graph=None,
                gpu_opt='GPU'):
        """__init__ [summary]
        
        Args:
            model_dir ([type], optional): [description]. Defaults to None.
            session ([type], optional): [description]. Defaults to None.
            graph ([type], optional): [description]. Defaults to None.
            gpu_opt (str, optional): [description]. Defaults to 'GPU'.
        
        Raises:
            ValueError: [description]
        """
        if model_dir is None:
            raise ValueError("No model directory provided.")
        else:
            self.meta_path, self.ckpt_path = self._get_model_path(model_dir)

        if (session is None) or (graph is None):
            self._sess, self._graph = self._get_sess_config()
        else:
            self._sess = session
            self._graph = graph

        if gpu_opt == "CPU":
            self._device = "/cpu:0"
        else:
            self._device = "/gpu:0"

        ### build inference graph.
        with self._graph.as_default():
            with tf.device(self._device):
                print("Load model from: ", self.meta_path)
                self._saver = tf.train.import_meta_graph(self.meta_path)
                self._saver.restore(self._sess, self.ckpt_path)

                self._model_graph = tf.get_default_graph()
                
                #basic node.
                self._inputs = None
                self._width = None
                self._training_mode = None
                self._logits = None


    def _get_model_path(self, model_dir):
        files = natsort.natsorted(os.listdir(model_dir))
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        meta_file = os.path.join(model_dir, meta_files[-1])

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            raise RuntimeError('No checkpoint file found')

        return meta_file, ckpt_path
    

    def _get_sess_config(self):
        tmp_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=tmp_graph)
        return sess, tmp_graph


    def inference(self,image_list):
        """inference: Core interface -- receive a list of images and return the recognize result.
        
        Args:
            image_list (array): a list of input images.
        
        Raises:
            NotImplementedError: if the method is not overloaded.
        """
        raise NotImplementedError("[inference] method is not implemented.")


    def greedy_search_decoder(self, logits, charset, merge_repeat=True):
        """greedy_search_decoder: dispatched.

        Args:
            logits (array): The RNN logits output. The shape is
                            [TimeSteps, 1, LengthOfCharSet].
                            Each timestep decodes an output.
            charset (list): List that restore the dict.
            merge_repeat (bool):

        Returns:
            decode_result: [description]
            decode_prob: [description]
        """
        decode_result = []
        decode_prob = []
        if_nerghbor = True
        for ts in range(logits.shape[0]):
            step_result = logits[ts][0]
            step_idx = nb_argmax(step_result)
            step_prob = math.exp(step_result[step_idx]) / nb_expsum(step_result)


            if step_idx == (len(charset) - 1):
                if_nerghbor = False
                continue
            if merge_repeat:
                if len(decode_result) == 0:
                    decode_result.append(charset[step_idx])
                    decode_prob.append(step_prob)
                    continue
                tmp_result = charset[step_idx]
                tmp_prob = step_prob
                if tmp_result != decode_result[-1]:
                    decode_result.append(tmp_result)
                    decode_prob.append(tmp_prob)
                elif not if_nerghbor:
                    decode_result.append(tmp_result)
                    decode_prob.append(tmp_prob)
                if_nerghbor = True
            else:
                decode_result.append(charset[step_idx])
                decode_prob.append(step_prob)

        return decode_result, decode_prob


    def _build_greedy_search_decoder(self,
                                    logits,
                                    width,
                                    merge_repeated=True):
        decode_out, decode_prob = tf.nn.ctc_greedy_decoder(logits, 
                                                        width,
                                                        merge_repeated=True)
        hypothesis = tf.cast(decode_out[0], tf.int32)

        dense_hyp = tf.sparse.to_dense(sp_input=hypothesis, default_value=-1)
        
        return dense_hyp,decode_prob


    def _build_beam_search_decoder(self,
                                    logits,
                                    width,
                                    beam_width=100,
                                    top_paths=1,
                                    merge_repeated=False):
        decode_out, decode_prob = tf.nn.ctc_beam_search_decoder(logits, 
                                                                width,
                                                                beam_width=beam_width,
                                                                top_paths=top_paths,
                                                                merge_repeated=False)
        hypothesis = tf.cast(decode_out[0], tf.int32)

        dense_hyp = tf.sparse.to_dense(sp_input=hypothesis, default_value=-1)
        
        return dense_hyp,decode_prob
    

    def _preprocess(self,image,image_height):
        raise NotImplementedError("[preprocess] method is not implemented.")


    def _predict_single(self,image):
        raise NotImplementedError("[predict_single] method is not implemented.")

    
    def _predict_batch(self,image_list):
        raise NotImplementedError("[predict_batch] method is not implemented.")
