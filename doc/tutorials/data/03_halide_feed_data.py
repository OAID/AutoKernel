#!/usr/bin/python3

import halide as hl
import numpy as np

w,h = 8,8

def addone():
    # feed input
    input_data = np.ones((4,4),dtype=np.uint8)
    A = hl.Buffer(input_data)

    i,j = hl.Var("i"), hl.Var("j")
    B = hl.Func("B")
    B[i,j] = A[i,j] + 1

    # output
    if 0:
        output = B.realize(4,4)
        print("out: \n",np.asanyarray(output))
    if 0:
        output = hl.Buffer(hl.UInt(8),[4,4])
        B.realize(output)
        print("out: \n",np.asanyarray(output))
    if 1:
        output_data = np.empty(input_data.shape, dtype=input_data.dtype,order="F")
        output = hl.Buffer(output_data)
        B.realize(output)
        print("out: \n",output_data)

addone()


