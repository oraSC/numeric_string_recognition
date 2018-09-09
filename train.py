import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 700
REGULARIZER = 0.0001
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
TRAIN_STEPS = 10000

MODEL_SAVE_PATH = './model'
MODEL_NAME = 'mnist_model'

def get_weight(shape , regularizer):
    w = tf.Variable(tf.truncated_normal(shape , stddev = 0.1))
    if regularizer !=None:
        tf.add_to_collection('losses' , tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b

def forward(input_tensor , average_class , regularizer ):
    # 定义神经网络参数
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    w2 = get_weight([LAYER1_NODE , OUTPUT_NODE] , regularizer)
    b2 = get_bias([OUTPUT_NODE])
    # 判断是否使用滑动平均
    if average_class == None :
        layer1 = tf.nn.relu(tf.matmul(input_tensor , w1) + b1)
        out_tensor = tf.matmul(layer1 , w2) + b2
    else :
        layer1 =tf.nn.relu(tf.matmul(input_tensor , average_class.average(w1)) + average_class.average(b1))
        out_tensor = tf.nn.relu(tf.matmul(layer1 , average_class.average(w2)) + average_class.average(b2))
    return out_tensor


def train(mnist):
    # 占位函数定义输入输出
    x = tf.placeholder(tf.float32 , [None , INPUT_NODE] , name = 'x-input')
    y_ = tf.placeholder(tf.float32 , [None , OUTPUT_NODE] , name = 'y-output')
    # 定义前向传播过程
    y = forward(x , None , REGULARIZER)
    # 定义反向传播过程
    ## 定义全局训练轮数张量（不可训练）
    global_step = tf.Variable(0 , trainable = False)
    ## 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(y_ , 1) , logits = y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    ## 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE , global_step , mnist.train.num_examples/BATCH_SIZE , LEARNING_RATE_DECAY , staircase = True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss , global_step = global_step)
    ## 定义滑动平均
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY , global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    train_op = tf.group(train_step , variable_average_op)
    # 定义一个saver
    saver = tf.train.Saver()
    # 开始训练
    with tf.Session() as sess:
        # 初始化计算图张量
        tf.global_variables_initializer().run()
        # 判断是否已有训练模型
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess , ckpt.model_checkpoint_path)
            # global_step= tf.Variable(eval(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) ,
            #                          trainable = False )
        for i in range(TRAIN_STEPS):
            xs , ys = mnist.train.next_batch(BATCH_SIZE)

            _ , loss_value , step = sess.run([train_op , loss , global_step] , feed_dict = {x:xs , y_:ys})
            if i%1000 ==0:
                print("after %d training step(s) , loss on training batch is %g"%(step , loss_value))
                # 保存模型
                saver.save(sess , os.path.join(MODEL_SAVE_PATH , MODEL_NAME) , global_step = global_step)

def main():
    mnist = input_data.read_data_sets("./data_sets" , one_hot = True)
    train(mnist)

if __name__ == '__main__':
    main()



