
#!/usr/bin/python3

import halide as hl

x, y = hl.Var("x"), hl.Var("y")
func = hl.Func("func")

func[x,y] = x + 10*y
#func.trace_stores()

out = func.realize(3, 4)  # width, height = 3,4

print("=============================")
for j in range(out.height()):
    for i in range(out.width()):
        print("out[x=%i,y=%i]=%i"%(i,j,out[i,j]))

print("Success!")


