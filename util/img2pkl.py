# Created by wz on 17-3-23.
# encoding=utf-8
import sys, cPickle
from scipy import misc
import numpy as np


def main():
    if len(sys.argv) < 2:
        print'eg: python img2pkl.py list.txt dst.pkl pardir\n' \
             'convert image to pickle\n' \
             'all the args are optional and all the imgs must have same size/channels\n\n'
        return
    l = len(sys.argv)
    src = sys.argv[1]
    dst = sys.argv[2] if l > 2 else 'data.pkl'
    par = sys.argv[3] if l > 3 else './pic'
    with open(src, 'r') as f:
        list = f.readlines()

    data = []
    labels = []
    for i in list:
        name, lable = i.strip('\n').split(' ')
        name = par + '/' + name
        print name + ' processed'
        img = misc.imread(name)
        img.resize((img.size, 1))
        data.append(img)
        labels.append(lable)

    print 'dump to pickle'
    with open(dst, 'wb') as f:
        cPickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)
    print 'completed'


if __name__ == '__main__':
    main()
