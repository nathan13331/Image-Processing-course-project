import pickle as pk
import numpy as np
import cv2
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        gz_dict = pk.load(fo, encoding='bytes')

    return gz_dict[b'labels'], gz_dict[b'filenames'], gz_dict[b'data']


for path in os.listdir(f'./datasets'):
    labels, names, datas = unpickle(f'./datasets/{path}')
    for label, name, data in zip(labels, names, datas):
        try:
            os.makedirs(f'pic/train/{label}')
            os.makedirs(f'pic/test/{label}')
        except:
            pass

        R = data[:1024]
        G = data[1024:2048]
        B = data[2048:3072]

        img, tmp = [], []
        for cnt, (r, g, b) in enumerate(zip(R, G, B), 1):
            tmp.append([b, g, r])
            if cnt % 32 == 0:
                img.append(tmp)
                tmp = []

        if path == 'test_batch':
            cv2.imwrite(f'pic/test/{label}/{str(name)[2:-1]}', np.array(img))
        else:
            cv2.imwrite(f'pic/train/{label}/{str(name)[2:-1]}', np.array(img))
