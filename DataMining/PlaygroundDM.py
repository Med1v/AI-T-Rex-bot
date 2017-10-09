import numpy as np

import cv2 as cv
from mss import mss
from PIL import Image

from pynput.keyboard import Key, Controller
from pynput import keyboard

data = []
dataSaved = False

# startGame = False
# gameOver = False
state = [False, False]  # startGame, gameOver

UP = False
DOWN = False

keyboardCont = Controller()

num_jumps = 0

mon = {'top': 425, 'left': 1200, 'width': 550, 'height': 100}
sct = mss()


def mouseHandler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print("L", x, y)
    elif event == cv.EVENT_RBUTTONDOWN:
        print("R", x, y)
        saveData()
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        print("mouse released")


def on_press(key):
    global UP, DOWN
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
        if key == Key.up:
            UP = True
        elif key == Key.down:
            DOWN = True
        # elif key == Key.shift:
        #     print("gameStart = {0}; gameOver = {1}" % state[0], state[1])


def on_release(key):
    global num_jumps
    print('{0} released'.format(
        key))
    global UP, DOWN
    if key == Key.up:
        UP = False
        state[0] = True
        num_jumps += 1
        print("jumps #", num_jumps)
    elif key == Key.down:
        DOWN = False

    if key == keyboard.Key.esc:
        # Stop listener
        return False


old_x_ = None


def writeData(UData, DData, KData, onlyJumps=False, past_inc=False,
              time_inc=False):
    global data, old_x_, num_jumps
    if KData[0] or KData[1]:
        KData.append(False)
    else:
        KData.append(True)
        if onlyJumps:
            return
    Knp = np.array(KData, dtype=bool)
    y_ = Knp * 1.0
    x_ = np.append(UData, DData)/255
    # print(x_, "size:", len(x_))
    # print("new data")

    if past_inc:
        if old_x_ is None:
            old_x_ = x_
        tmp = np.append(x_, old_x_)
        old_x_ = x_
        x_ = tmp

    if time_inc:
        gtime = num_jumps/130
        time_arr = [gtime for i in range(11)]
        # print(gtime)
        x_ = np.append(x_, time_arr)

    data.append((x_, y_))
    # print("data:\n", data)
    # print(ddir)
    return


def saveData(ddir='D:/Develop/TensorFlow/TRex/files/input_data', fid='u'):
    global data, dataSaved
    if not dataSaved:
        if len(data) < 21:
            print("Nothing to save")
            return
        # np.savez_compressed(ddir + '/eval/test_data_' + fid, data)
        print("Saving the data...")
        np.savez_compressed(ddir + '/train_data_' + fid, data[0:-40])
        print("file saved to ", ddir)
        dataSaved = True


def makeWider(arr, second=None, arrwidth=30):
    inpt = [arr]
    for i in range(0, arrwidth):
        inpt.append(arr)
    if second is not None:
        for i in range(0, arrwidth):
            inpt.append(second)
    return np.array(inpt)


def loop(conn=state):
    global UP, DOWN
    cv.namedWindow('test')
    cv.setMouseCallback('test', mouseHandler)

    while 1:
        im = sct.grab(mon)
        img = Image.frombytes('RGB', im.size, im.rgb).convert('L')

        # print(np.array(img)[40][242])
        if np.array(img)[5][199] > 100:
            # print("not gg")
            conn[1] = False
        else:
            # print("gg")
            conn[1] = True

        playGroundU = np.array(img)[64]
        playGroundB = np.array(img)[80]

        # cv.imshow('test', makeWider(playGroundU, second=playGroundB))
        cv.imshow('test', np.array(img))
        # print(conn[0])
        if not conn[1] and conn[0]:
            writeData(playGroundU, playGroundB, [UP, DOWN], time_inc=True)
        if conn[1] and conn[0]:
            saveData(fid='99')

        # keyboardCont.press('q')

        k = cv.waitKeyEx(25)
        if k & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    # Collect events until released
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    # ml = Process(target=loop, args=(
    #              'D:/Develop/TensorFlow/TRex/files/input_data',
    #              stateArr))
    # ml.start()
    loop()
