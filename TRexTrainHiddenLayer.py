import tensorflow as tf
import numpy as np

import os
import os.path

import cv2 as cv
from mss import mss
from PIL import Image

from pynput.keyboard import Key, Controller

mon = {'top': 425, 'left': 1200, 'width': 550, 'height': 100}
sct = mss()

keyboardCont = Controller()

fullData = None
not_printed = True


def readBatch(num, deval=False):
    global fullData, not_printed
    path = 'D:/Develop/TensorFlow/TRex/files/input_data/'
    if deval:
        path = 'D:/Develop/TensorFlow/TRex/files/input_data/eval/'

    if fullData is None:
        print("Reading data")
        dir_files = [name for name in os.listdir(path)
                     if os.path.isfile(os.path.join(path, name))]
        arr = []
        for fn in dir_files:
            tmp = np.load(path + fn)
            arr.append(tmp[tmp.files[0]])

        fullData = np.concatenate(arr)

    dataSize = len(fullData)

    if not_printed:
        print("data size:", dataSize)
        not_printed = False
    # return
    randID = np.round(np.random.rand(num) * (dataSize - 1)).astype(int)
    # print(randID)

    x_arr = np.zeros((num, 1100))
    y_arr = np.zeros((num, 3))

    # print(len(fullData[randID[0]][0]))
    # return

    for i in range(num):
        x_arr[i] = fullData[randID[i]][0]
        y_arr[i] = fullData[randID[i]][1]

    res = (x_arr, y_arr)
    # print(res)

    return res


def makeWider(arr, second=None, arrwidth=30):
    inpt = [arr]
    for i in range(0, arrwidth):
        inpt.append(arr)
    if second is not None:
        for i in range(0, arrwidth):
            inpt.append(second)
    return np.array(inpt)


old_inpt = None


def main(param):
    global old_inpt
    # Create the model
    x = tf.placeholder(tf.float32, [None, 1100])
    W_1 = tf.Variable(tf.zeros([1100, 220]))
    W_2 = tf.Variable(tf.zeros([220, 3]))
    b = tf.Variable(tf.zeros([3]))

    h = tf.nn.sigmoid(tf.matmul(x, W_1))

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_drop = tf.nn.dropout(h, keep_prob)

    y = tf.matmul(h_drop, W_2) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # train_step =
    # tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    path = "D:/Develop/TensorFlow/TRex/files/lasthl/TRex.brain"
    # 0 - train; 1 - evaluate; 9 - lounch loop
    param = 9  # CHANGE PATH
    # print(param)

    # range 10500 batch 12 prob 0.55
    # Train
    if param == 0:
        print("Start TRex training!")
        # Train
        for i in range(10400):
            batch_xs, batch_ys = readBatch(12)
            if i % 100 == 0:
                print('step %d, training accuracy %g' %
                      (i, sess.run(accuracy,
                                   feed_dict={x: batch_xs, y_: batch_ys,
                                              keep_prob: 1})))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,
                                            keep_prob: 0.55})

        # Test trained model
        testing_data = readBatch(500, deval=True)
        print(sess.run(accuracy, feed_dict={x: testing_data[0],
                                            y_: testing_data[1],
                                            keep_prob: 1}))

        save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)

    # Evaluate
    elif param == 1:
        saver.restore(sess, path)

        # Test trained model
        testing_data = readBatch(5000, deval=True)
        print(sess.run(accuracy, feed_dict={x: testing_data[0],
                                            y_: testing_data[1],
                                            keep_prob: 1}))

    # lounch loop
    elif param == 9:
        saver.restore(sess, path)
        prediction = tf.argmax(y, 1)

        cv.namedWindow('test')
        count_jump = True
        jumps = 0
        shift_speed = 1
        lshift_acc = mon['left']
        print("Start")
        while 1:
            if jumps%10 == 0:
                jumps += 1
                lshift_acc += shift_speed
                if lshift_acc > 1335:
                    lshift_acc = 1340
                shift_speed += 1
                if shift_speed > 15:
                    shift_speed = 15
                mon['left'] = round(lshift_acc)
                print("screen moved right, now it's:", mon['left'])
            im = sct.grab(mon)
            img = Image.frombytes('RGB', im.size, im.rgb).convert('L')
            playGroundU = np.array(img)[64]  # 64
            playGroundB = np.array(img)[80]
            inpt = np.append(playGroundU, playGroundB)

            # For big data (2200)
            # if old_inpt is None:
            #     old_inpt = inpt
            # tmp = np.append(inpt, old_inpt)
            # old_inpt = inpt
            # inpt = tmp

            # print("inpt =", np.array([inpt]))
            cv.imshow('test', makeWider(playGroundU, second=playGroundB))
            # cv.imshow('test', np.array(img))
            result = sess.run(prediction, feed_dict={x: np.array([inpt]),
                                                     keep_prob: 1})
            # print(result[0])
            if result[0] == 0:
                keyboardCont.release(Key.down)
                keyboardCont.press(Key.up)
                if count_jump:
                    jumps += 1
                    count_jump = False
                # print("UP")
            # elif result[0] == 1:
            #     keyboardCont.press(Key.down)
                # print("DOWN")
            else:
                count_jump = True
                # keyboardCont.press(Key.down)
            # print("")

            # keyboardCont.press('q')
            k = cv.waitKeyEx(25)
            if k & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break

        # run_data = readBatch(1)[0]
        # print(run_data)
        # result = sess.run(prediction, feed_dict={x: run_data})
        # print(result[0])


if __name__ == '__main__':
    # res = readBatch(5)
    # print(res)
    tf.app.run(main=main)
