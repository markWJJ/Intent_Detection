import numpy as np

# s1=np.array([[1,2,3],[3,4,5]])
#
# s2=np.tile(s1,(2,1))
#
# s3=np.repeat(s1,60,axis=0)
#

# print(s3.shape)

from collections import Counter,OrderedDict

s=[1,1,1,22,2,2,3,3,3]

ss=[[k,v] for k,v in Counter(s).items()]
ss.sort(key=lambda x:x[1],reverse=True)

print(ss[0][0])
