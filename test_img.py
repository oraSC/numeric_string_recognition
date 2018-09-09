import os

import cv2
from matplotlib import pyplot as plt
from libs.helper import search_numeric_strings


class Numeric_String:

    def __init__(self , img ,numericstring_regions ):
        self.total_num = len(numericstring_regions)
        self.number_img = []
        fig , axes = plt.subplots(ncols = 1 , nrows =  self.total_num)
        for index , region in enumerate(numericstring_regions) :
            x, y, w, h = (region[0], region[1], region[2], region[3])
            top = int(y - h/4) if int(y - h/4) > 0 else 1
            bottom = int(y + h*1.25) if int(y + h*1.25) < img.shape[0] else img.shape[0] - 1
            left = int(x - w/4) if int(x - w/4) > 0 else 1
            right = int(x + w*1.25) if int(x + w*1.25) < img.shape[1] else img.shape[1] - 1
            mask_img = img[ top : bottom ,left : right ]
            self.number_img.append(mask_img)
            axes[index].imshow(mask_img)
        plt.show()





class Test_Img:

    def __init__(self):
        self.total = 0
        self.numeric_strings = []

    def fill(self , img):
        numeric_strings_regions = search_numeric_strings(img)
        self.total = len(numeric_strings_regions)
        self.numeric_strings = [Numeric_String(img , numericstring_regions)  for numericstring_regions in numeric_strings_regions]

if __name__ == '__main__':
    project_path = os.getcwd()
    img_path = os.path.join(project_path, './test_pic/10.png')
    img = cv2.imread(img_path)
    test_img = Test_Img()
    test_img.fill(img)

    for num_string in test_img.numeric_strings:
        for num_img in num_string.number_img:
            cv2.imshow('1' , num_img)
            cv2.waitKey(0)
