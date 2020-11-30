
#!/usr/bin/python3

import halide as hl

x, y = hl.Var("x"), hl.Var("y")
w,h = 8,8

def origin():
    func = hl.Func("func_origin")
    func[x,y] = x + 10*y
    out = func.realize(w, h)
    func.print_loop_nest()

    '''
    运行结果：
    produce func_origin:
    for y:
      for x:
        func_origin(...) = ...
    '''
    print("---------------------")

def reorder():
    func = hl.Func("func_reorder")
    func[x,y] = x + 10*y
    func.reorder(y,x)
    out = func.realize(w, h)
    func.print_loop_nest()

    '''
    运行结果：
    produce func_reorder:
    for x:
      for y:
        func_reorder(...) = ...
    '''
    print("---------------------")

def split():
    func = hl.Func("func_split")
    func[x,y] = x + 10*y

    xo, xi = hl.Var("x_outer"), hl.Var("x_inner")
    func.split(x, xo, xi,4)
    out = func.realize(w, h)
    func.print_loop_nest()
    '''
    运行结果：
    produce func_split:
    for y:
      for x.x_outter:
        for x.x_inner in [0, 3]:
          func_split(...) = ...
    '''
    print("---------------------")

def fuse():
    func = hl.Func("func_fuse")
    func[x,y] = x + 10*y

    xy_fuse = hl.Var("xy_fuse")
    func.fuse(x,y,xy_fuse)
    out = func.realize(w, h)
    func.print_loop_nest()
    '''
    运行结果：
    produce func_fuse:
      for x.xy_fuse:
        func_fuse(...) = ...
    '''
    print("---------------------")

def tile():
    func = hl.Func("func_tile")
    func[x,y] = x + 10*y

    xo, xi, yo, yi = hl.Var("xo"), hl.Var("xi"),hl.Var("yo"), hl.Var("yi")
    xfactor, yfactor = 4, 8
    func.tile(x,y,xo,yo,xi,yi,xfactor,yfactor)
    out = func.realize(w, h)
    func.print_loop_nest()
    '''
    运行结果：
    produce func_tile:
    for y.yo:
      for x.xo:
        for y.yi in [0, 7]:
          for x.xi in [0, 3]:
            func_tile(...) = ...
    '''
    print("---------------------")

def vectorize():
    func = hl.Func("func_vectorize")
    func[x,y] = x + 10*y

    factor = 4
    func.vectorize(x,factor)
    out = func.realize(w, h)
    func.print_loop_nest()
    '''
    运行结果：
    produce func_vectorize:
    for y:
      for x.x:
        vectorized x.v0 in [0, 3]:
          func_vectorize(...) = ...`
    '''
    print("---------------------")

def unroll():
    func = hl.Func("func_unroll")
    func[x,y] = x + 10*y

    factor = 2
    func.unroll(x,factor)
    out = func.realize(w, h)
    func.print_loop_nest()
    '''
    运行结果：
    produce func_unroll:
    for y:
      for x.x:
        unrolled x.v1 in [0, 1]:
          func_unroll(...) = ...
    '''
    print("---------------------")

def parallel():
    func = hl.Func("func_parallel")
    func[x,y] = x + 10*y

    factor = 4
    func.parallel(x,factor)
    out = func.realize(w, h)
    func.print_loop_nest()
    '''
    运行结果：
    produce func_parallel:
    for y:
      parallel x.x:
        for x.v2 in [0, 3]:
          func_parallel(...) = ...
    '''
    print("---------------------")



def default_inline():
    print("=" * 50)
    x, y = hl.Var("x"), hl.Var("y")
    A, B = hl.Func("A_default"), hl.Func("B_default")
    A[x, y] = x + 10*y
    B[x, y] = A[x, y] + 1

    print("pipeline with default schedule: inline")
    print('-'*50)
    B.realize(w, h)
    B.print_loop_nest()

def compute_at():
    print("=" * 50)
    A, B = hl.Func("A_y"), hl.Func("B_y")
    A[x, y] = x + 10*y
    B[x, y] = A[x, y] + 1
    
    print("pipeline with schedule: A.compute_at(B,y)")
    print('-'*50)
    A.compute_at(B, y)
    B.realize(w, h)
    B.print_loop_nest()

def compute_root():
    print("=" * 50)
    A, B = hl.Func("A_root"), hl.Func("B_root")
    A[x, y] = x + 10*y
    B[x, y] = A[x, y] + 1
    
    print("pipeline with schedule: A.compute_root()")
    print('-'*50)
    A.compute_root()
    B.realize(w, h)
    B.print_loop_nest()
'''
==================================================
pipeline with default schedule: inline
--------------------------------------------------
produce B_default:
  for y:
    for x:
      B_default(...) = ...
==================================================
pipeline with schedule: A.compute_at(B,y)
--------------------------------------------------
produce B_y:
  for y:
    produce A_y:
      for x:
        A_y(...) = ...
    consume A_y:
      for x:
        B_y(...) = ...
==================================================
pipeline with schedule: A.compute_root()
--------------------------------------------------
produce A_root:
  for y:
    for x:
      A_root(...) = ...
consume A_root:
  produce B_root:
    for y:
      for x:
        B_root(...) = ...
'''

origin()
reorder()
split()
fuse()
tile()
vectorize()
unroll()
parallel()

default_inline()
compute_at()
compute_root()

print("Success!")

