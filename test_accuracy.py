

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from train import forward, REGULARIZER, INPUT_NODE, OUTPUT_NODE, MOVING_AVERAGE_DECAY, MODEL_SAVE_PATH


def test_acc(mnist):
    # 清除默认图的堆栈，并设置全局图为默认图
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')
    y = forward(x, None, REGULARIZER)

    # 加载滑动平均参数到神经网络参数
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)

    # 定义测试准确率函数
    correct_prediction = tf.equal(tf.argmax(y , 1) , tf.argmax(y_ , 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
        else:
            print('No checkpoint file found')
            return





def main():
    mnist = input_data.read_data_sets("./data_sets", one_hot=True)
    test_acc(mnist)


if __name__ == '__main__':
    main()