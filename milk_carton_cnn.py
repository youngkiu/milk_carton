import tensorflow as tf
import random
import os
import numpy as np

from PIL import Image


def load_train_data():
    normal_img_dir = "skive/normal_img/"
    defect_img_dir = "skive/defect_img/"

    train_x = []
    train_y = []

    files = os.listdir(normal_img_dir)
    for file in files:
        filename = "{0}{1}".format(normal_img_dir, file)
        img = Image.open(filename).convert("L")
        pix = np.array(img)

        if train_x == []:
            train_x = [np.concatenate(pix)]
            train_y = [[1, 0]]
        else:
            train_x = np.concatenate((train_x, [np.concatenate(pix)]), axis=0)
            train_y = np.concatenate((train_y, [[1, 0]]), axis=0)

    files = os.listdir(defect_img_dir)
    for file in files:
        if file.find("image") >= 0:
            filename = "{0}{1}".format(defect_img_dir, file)
            img = Image.open(filename).convert("L")
            pix = np.array(img)

            train_x = np.concatenate((train_x, [np.concatenate(pix)]), axis=0)
            train_y = np.concatenate((train_y, [[0, 1]]), axis=0)

    return train_x, train_y

def load_test_data():
    test_img_dir = "skive/test/"

    test_x = []
    test_file = []

    files = os.listdir(test_img_dir)
    for file in files:
        filename = "{0}{1}".format(test_img_dir, file)
        img = Image.open(filename).convert("L")
        pix = np.array(img)

        if test_x == []:
            test_x = [np.concatenate(pix)]
            test_file = [filename]
        else:
            test_x = np.concatenate((test_x, [np.concatenate(pix)]), axis=0)
            test_file = np.concatenate((test_file, [filename]), axis=0)

    return test_x, test_file

if __name__ == '__main__':
    tf.set_random_seed(20170429)  # reproducibility

    train_x, train_y = load_train_data()
    print(train_x.shape)
    print(train_y.shape)
    
    num_of_train = np.size(train_x, 0)
    image_size = np.size(train_x, 1)
    nb_classes = np.size(train_y, 1)

    # hyper parameters
    learning_rate = 0.0001
    training_epochs = 15
    batch_size = 3

    # input place holders
    X = tf.placeholder(tf.float32, [None, image_size])
    X_img = tf.reshape(X, [-1, 1660, 300, 1])   # img 1660x300x1 (black/white)
    Y = tf.placeholder(tf.float32, [None, nb_classes])

    # C1 ImgIn shape=(?, 1660, 300, 1)
    W1 = tf.Variable(tf.random_normal([5, 5, 1, 4], stddev=0.01))
    #    Conv     -> (?, 1660, 300, 4)
    #    Pool     -> (?, 830, 150, 4)
    C1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    C1 = tf.nn.relu(C1)
    S2 = tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    '''
    Tensor("Conv2D:0", shape=(?, 1660, 300, 4), dtype=float32)
    Tensor("Relu:0", shape=(?, 1660, 300, 4), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 830, 150, 4), dtype=float32)
    '''

    # C3 ImgIn shape=(?, 830, 150, 4)
    W2 = tf.Variable(tf.random_normal([5, 5, 4, 8], stddev=0.01))
    #    Conv      ->(?, 830, 150, 8)
    #    Pool      ->(?, 415, 75, 8)
    C3 = tf.nn.conv2d(S2, W2, strides=[1, 1, 1, 1], padding='SAME')
    C3 = tf.nn.relu(C3)
    S4 = tf.nn.max_pool(C3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    F5 = tf.reshape(S4, [-1, 415 * 75 * 8])
    '''
    Tensor("Conv2D_1:0", shape=(?, 830, 150, 8), dtype=float32)
    Tensor("Relu_1:0", shape=(?, 830, 150, 8), dtype=float32)
    Tensor("MaxPool_1:0", shape=(?, 415, 75, 8), dtype=float32)
    Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
    '''

    # Final FC 415x75x8 inputs -> nb_classes outputs
    W3 = tf.get_variable("W3", shape=[415 * 75 * 8, nb_classes],
                         initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([nb_classes]))
    logits = tf.matmul(F5, W3) + b

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning started. It takes sometime.')

    total_batch = int(num_of_train / batch_size)
    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_y = train_y[i*batch_size:(i+1)*batch_size]
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model
    test_x, test_file = load_test_data()
    print(test_x.shape)

    num_of_test = np.size(test_x, 0)
    for i in range(num_of_test):
        prediction = sess.run(tf.argmax(logits, 1), feed_dict={X: test_x[i:i+1]})
        print("{0} \t{1}".format(test_file[i], "normal" if prediction == 0 else "defect"))

'''
(153, 498000)
(153, 2)
Learning started. It takes sometime.
Epoch: 0001 cost = 3.529674595
Epoch: 0002 cost = 1.198152409
Epoch: 0003 cost = 0.746434966
Epoch: 0004 cost = 0.918099420
Epoch: 0005 cost = 0.348275243
Epoch: 0006 cost = 0.175957333
Epoch: 0007 cost = 0.250059961
Epoch: 0008 cost = 0.256329637
Epoch: 0009 cost = 0.159760816
Epoch: 0010 cost = 0.085794412
Epoch: 0011 cost = 0.077970891
Epoch: 0012 cost = 0.068665664
Epoch: 0013 cost = 0.061409960
Epoch: 0014 cost = 0.053886106
Epoch: 0015 cost = 0.056859106
Learning Finished!
(19, 498000)
skive/test/0.png 	normal
skive/test/1.png 	normal
skive/test/10.png 	defect
skive/test/11.png 	defect
skive/test/12.png 	defect
skive/test/13.png 	defect
skive/test/14.png 	defect
skive/test/15.png 	defect
skive/test/16.png 	defect
skive/test/17.png 	defect
skive/test/18.png 	defect
skive/test/2.png 	normal
skive/test/3.png 	normal
skive/test/4.png 	normal
skive/test/5.png 	normal
skive/test/6.png 	normal
skive/test/7.png 	normal
skive/test/8.png 	normal
skive/test/9.png 	normal
'''
