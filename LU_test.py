#! /usr/bin/env python3
# author bravomikekilo bravomikekilo@buaa.edu.cn
# under GPLv3
# for LICENSE see LICENSE at root.
import numpy as np
import LU

a = np.array([[8.1, 2.3, -1.5], [0.5, -6.23, 0.87], [2.5, 1.5, 10.2]])
print('input|>')
print(a)

L, U = LU.crout(a, inplace=False)
print('L|>')
print(L)
print('U|>')
print(U)
print('LU|>')
print(L.dot(U))

A = np.array([[1,2,3],[4,3,6],[7,8,9]], dtype=np.float32)
b = np.array([8,7,5], dtype=np.float32).T

print('A|>\n', A, sep='')
print('b|>\n', b, sep='')
x = LU.LU_solve(A.copy(), b)
print('x|>\n', x, sep='')
print("b'|>\n", A.dot(x), sep='')