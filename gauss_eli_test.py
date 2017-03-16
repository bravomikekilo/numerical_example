#! /usr/bin/env python3
# author bravomikekilo bravomikekilo@buaa.edu.cn
# under GPLv3
# for LICENSE see LICENSE at root.
import numpy as np
import gauss_eli

a = np.array([[1,2,3,8],[4,3,6,7],[7,8,9,5]], dtype=np.float32)
print(a.shape)
print('input|>')
print(a)

print("output of common eli|>")
re = gauss_eli.gauss_eli(a.copy())
print(re)
ret = gauss_eli.back_iter(re).T
print(ret)
A = a[:, :a.shape[0]]
print(A.dot(ret))

print('output of column major eli|>')
re = gauss_eli.col_major_gauss_eli(a.copy())
print(re)
ret = gauss_eli.back_iter(re).T
print(ret)
A = re[:, :a.shape[0]]
print(A.dot(ret))

A = np.array([[1,2,3],[4,3,6],[7,8,9]], dtype=np.float32)
b = np.array([8,7,5], dtype=np.float32).T
print('direct solve|>')
print(gauss_eli.solve(A, b))