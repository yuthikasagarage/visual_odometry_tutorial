import torch
import cv2  
import numpy as np


print(cv2.__version__ )

print(torch.has_mps) 

nv = np.array([[1,2,3],]) 

nvT = nv.T 

# print(nvT, nv) 

nm = np.array([ [1,2,3,4],
                [4,5,6,7] ]) 


nmT = nm.T 

# print(nmT, nm) 

# using torch


tv = torch.tensor([[1,2,3,4], [3,4,5,6]])

tvT = tv.T

# print(tvT, tv) 

# dot product
# only for similar dimensional vectors 

#for matrices 1 st column count should match to 2nd row count in order to dot the two matrices
# mr nice 

nm1 =  np.random.rand(3,2) # 3 rows 2 columns

nm2 = np.random.rand(2,4) # 2 rows 4 columns

print(np.dot(nm1, nm2))

# print(nm1, nm2)


tv1 = torch.tensor([1,2,3])
tv2 = torch.tensor([3,4,5])

print(torch.dot(tv1, tv2) , '  ', type(tv1), type(tv2))


# multipled matrix is [1 st matrix row size , 2 matrix column size]


A = np.random.rand(3,4) 
B = np.random.rand(4,5) 

print( np.round(A@B, 2)) 
print ( np.round(np.matmul(A, B), 2 ))


A = torch.randn(3,5) 
B = torch.randn(5,6) 
C1 = np.random.rand(5,6)
C2 = torch.tensor(C1, dtype=torch.float)


print (np.round(A@C2, 2))


# cross entropy binary, shannon entropy
import torch.nn.functional as F

p = [   1,0   ] # sum=1
q = [ .25,.75 ] # sum=1


q_tensor = torch.Tensor(q) 
p_tensor = torch.Tensor(p)

print(F.binary_cross_entropy(q_tensor,p_tensor))


v = np.array([1, 3,4,34])

minval = np.min(v)


print(minval, np.max(v), np.argmax(v), np.argmin(v))


M = np.array([[1,2,3], [2,34,534]])

print(np.min(M, axis = 1))