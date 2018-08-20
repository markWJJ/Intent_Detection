# pop_list=[1,2,3,4,5]
#
# push_list=[4,5,3,1,2]
# length=len(pop_list[:])
# stack=[]
# for i in range(2*length):
#     if i<=length:
#         stack.append(pop_list[i])
#
#
#     if push_list.index(stack[-1])==0:
#         print(stack[-1])
#         push_list=push_list[1:]
#     pop_list=pop_list[:-1]

import numpy as np


s=np.zeros(shape=(3,))
s1=np.ones(shape=(3,))


ss=np.sum(np.equal(s,s1))

print(ss)