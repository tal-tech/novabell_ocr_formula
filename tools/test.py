# code: utf-8
import os 
import sys
import time
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
import interface
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../detector'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../recognizor'))

class test_interface(object):


    def __init__(self):
        self.text_interpreter = interface.TextInterpreter()

        self.test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../data/2019-05-14-06-44-39.png")


    def run_single_image(self):
        tmp_img = cv2.imread(self.test_image_path)

        while True:
            ipt_result,proc_result = self.text_interpreter.interpret(tmp_img)
            print(ipt_result)
            print(proc_result)


if __name__ == "__main__":
    tester = test_interface()
    tester.run_single_image()



