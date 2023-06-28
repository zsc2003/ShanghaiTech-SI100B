import numpy as np
a=np.array([1,2,3,4,5,6,7,1,2,3,4,5])
# b=a==1
# print(b)
c=np.array([0,0,0,0,0,0,0,0])
c[[1,1,2,3]]+=[8,9,8,7]
c[[1,1,2,3]]=[8,9,8,7]
print(c)