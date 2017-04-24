# Created by wz on 17-3-24.
# encoding=utf-8
import cPickle
import numpy as np

with open('../data/swjtu_verif.pkl') as f:
    obj = cPickle.load(f)
a, b, c = obj
print a[0][0].dtype
print len(a)
aa = np.array(map(lambda x: x[0], a), dtype=np.uint8)
aalabel = np.array(map(lambda x: np.argmax(x[1]), a), dtype=np.uint8)
print type(aa), aa.dtype, aa.shape
with open('../data/train.pkl', 'wb') as f:
    cPickle.dump({'data': aa, 'label': aalabel}, f)

aa = np.array(map(lambda x: x[0], b), dtype=np.uint8)
aalabel = np.array(map(lambda x: np.argmax(x[1]), b), dtype=np.uint8)
with open('../data/validate.pkl', 'wb') as f:
    cPickle.dump({'data': aa, 'label': aalabel}, f)

aa = np.array(map(lambda x: x[0], c), dtype=np.uint8)
aalabel = np.array(map(lambda x: np.argmax(x[1]), b), dtype=np.uint8)
with open('../data/test.pkl', 'wb') as f:
    cPickle.dump({'data': aa, 'label': aalabel}, f)
