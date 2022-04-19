import matplotlib.pyplot as plt 
import numpy as np
from tools.utils import *

factor=2

step = np.array((-4,-10))

A = np.random.normal(loc=(15,6),scale=(2,1),size=(60,2))
B = factor*(np.array((A[:,1],A[:,0])) ).T
B[:,0] = A[:,1].mean()+ (B[:,0] - B[:,0].mean())
B = B+step

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.scatter(A.T[0],A.T[1], label='raw A')
ax.scatter(B.T[0],B.T[1], label='raw B')
plt.axis('equal')
plt.title(f'Raw test data, B scaled with factor {factor} \nand moved by xy step {step}.')
plt.legend()
plt.savefig('img_OP/OPtest_raw.png',dpi=100,transparent=False)

std_func = centerunit

A_std = std_func(A)
B_std = std_func(B)
ab_align = OrthogProcrustAlign(A_std,B_std,standard=True)
C = B_std.dot(ab_align)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.scatter(A_std.T[0],A_std.T[1],lw=4,label='standard A')
ax.scatter(B_std.T[0],B_std.T[1],label='standard B')
ax.scatter(C.T[0],C.T[1],lw=1,alpha=0.5,c='yellow', label='aligned standard B')
plt.title(f'Aligned test data using *{std_func.__name__}*')
plt.legend()
plt.savefig('img_OP/OPtest_aligned_{}.png'.format(std_func.__name__),dpi=100,transparent=False)

