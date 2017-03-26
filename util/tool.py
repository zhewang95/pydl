# Created by wz on 17-3-24.
# encoding=utf-8
import cPickle
import numpy as np

with open('../data/swjtu_verif.pkl') as f:
    obj=cPickle.load(f)
a,b=obj[:2]
print a[0][0].dtype
aa=np.array(map(lambda x:x[0],a))-0.5
aalabel=np.array(map(lambda x:np.argmax(x[1]),a))
print type(aa),aa.dtype,aa.shape
with open('../data/test1.pkl','wb') as f:
    cPickle.dump({'data':aa,'labels':aalabel},f)


aa=np.array(map(lambda x:x[0],b))-0.5
aalabel=np.array(map(lambda x:np.argmax(x[1]),b))
with open('../data/test2.pkl','wb') as f:
    cPickle.dump({'data':aa,'labels':aalabel},f)
