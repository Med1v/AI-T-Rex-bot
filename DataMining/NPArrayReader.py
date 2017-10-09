import numpy as np

fpath = 'D:/Develop/TensorFlow/TRexTwoOh/recorded_data/numbers_0'
arr = np.load(fpath + '.npz')

print("files = ", arr.files)
# print(arr[arr.files[0]])
data = arr[arr.files[0]]
print(data)
# print(len(arr[arr.files[0]][0][0]))

# useless. Used for failed improvements
def change_data(data):
    if len(data[0][0]) > 1220:
        print("file already changed")
        return
    for i in range(len(data)):
        x = data[i][0]
        data[i][0] = np.append(x, [x[-1] for i in range(110)])

    print(len(data))
    print(len(data[0][0]))

    print("Saving the data...")
    np.savez_compressed(fpath, data)
    print("file saved to ", fpath)


# change_data(arr[arr.files[0]])
