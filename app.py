import os
import cv2
from matplotlib import  pyplot as plt
import numpy as np
import tensorflow as tf

from test_img import Test_Img
from train import INPUT_NODE, forward, REGULARIZER, MOVING_AVERAGE_DECAY, MODEL_SAVE_PATH



def image_pre(img):
    img = cv2.resize(img , (28,28))
    plt.figure('1'), plt.subplot(332), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('resize')
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    plt.figure('1'), plt.subplot(333), plt.imshow(img_gray, 'gray'), plt.title('gray')
    ret , img_threshold = cv2.threshold(img_gray , 0 ,255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure('1'), plt.subplot(334), plt.imshow(img_threshold, 'gray'), plt.title('img_threshold')
    img_not = cv2.bitwise_not(img_threshold)
    # img_not = img_threshold
    plt.figure('1'), plt.subplot(335), plt.imshow(img_not, 'gray'), plt.title('img_not')
    plt.show()
    img_arr = img_not.reshape([1 , 784])
    img_arr = img_arr.astype(np.float)
    img_arr = np.multiply(img_arr , 1.0/255.0)
    return img_arr

def predict(img_array):
    # 清除默认图的堆栈，并设置全局图为默认图
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32 , [None , INPUT_NODE] , name = 'x-input')
    y = forward(x , None , REGULARIZER)
    preValue = tf.argmax(y, 1)
    # 加载滑动平均参数到神经网络参数
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)

    with tf.Session() as sess :
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess , ckpt.model_checkpoint_path)
            result = sess.run(preValue , feed_dict = {x :img_array})
            return result
        else :
            print('No model found')
            return -1



def app(img):
    test_img = Test_Img()
    test_img.fill(img)
    for num_string in test_img.numeric_strings:
        for num_img in num_string.number_img:
            cv2.imshow('1' , num_img)
            cv2.waitKey(0)
            img_array = image_pre(num_img)
            result = predict(img_array)
            print("The prediction number is:", result)


def get_img():
    # img_num = eval(input('input the number of test image :'))
    # img_path = input('input the path of image :')
    pass

def main():
    img_num = 1
    for i in range(img_num):
        img_origin = cv2.imread('./test_pic/0.png' )
        app(img_origin)
        plt.figure('1'),plt.subplot(331),plt.imshow(cv2.cvtColor(img_origin , cv2.COLOR_BGR2RGB)) ,plt.title('origin')
        plt.show()


if __name__ == '__main__':
    main()
