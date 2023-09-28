from detector.cut_chinese_math.nets import cnn,cnn_64


net_dict = {
    'cnn':cnn,
    'cnn_64':cnn_64
}





def get_net(name):
    return net_dict[name].Model()