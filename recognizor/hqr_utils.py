# coding:utf-8

import os
import sys
import numpy as np
# from hqr_config import CONFIG 
import yaml


config_file = open(os.path.dirname(os.path.abspath(__file__))+'/hqr_config.yaml', encoding='utf-8')
CONFIG = yaml.load(config_file,Loader=yaml.FullLoader)
config_file.close()


def get_config():
    return CONFIG


def _get_CharSet_list(charset_dir=CONFIG["CHARSET_FILE"]):
    print(charset_dir)
    dict_file = open(charset_dir, 'r' , encoding='utf-8')
    CharSet = []
    for line in dict_file:
        tmp_char = line.strip().split()[1]
        CharSet.append(tmp_char)
    return CharSet


def _get_CharSet_dict(charset_dir=CONFIG["CHARSET_FILE"]):
    dict_file = open(charset_dir, 'r', encoding='utf-8')
    CharSet = {}
    for line in dict_file:
        tmp_line = line.strip().split()
        tmp_label = int(tmp_line[0])
        tmp_char = tmp_line[1]
        tmp_count = int(tmp_line[3])
        CharSet[tmp_char] = tmp_label
    return CharSet


CharSet = _get_CharSet_list()
CharSet_dict = _get_CharSet_dict()


def get_CharSet_dict():
    return CharSet_dict


def get_CharSet_list():
    return CharSet


def generate_CharSet_file():
    pass

def save_config(save_dir):
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding='utf-8') as yaml_file:
        yaml.dump(CONFIG, yaml_file)


if __name__ == "__main__":
    # test get_CharSet_list()
    get_CharSet_list()

    # test get_CharSet_dict()
    get_CharSet_dict()
    
    # test generate_CharSet_file()