
#!/usr/bin/python3

import halide as hl
import time

def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        fn(*args, **kwargs)
        print("%s cost %.6f second"%(fn.__name__, float(time.clock() - start)))
    return _wrapper


x, y = hl.Var("x"), hl.Var("y")

@time_me
def func_origin__(w,h):
    func = hl.Func("func")
    func[x,y] = x + 10*y
    out = func.realize(w, h)

@time_me
def func_parallel(w,h):
    func = hl.Func("func")
    func[x,y] = x + 10*y
    func.parallel(y,4)
    func.realize(w,h)


func_origin__(400,400)
func_parallel(400,400)

'''
运行结果：
func_origin__ cost 0.510215 second
func_parallel cost 0.122265 second
'''

print("Success!")

