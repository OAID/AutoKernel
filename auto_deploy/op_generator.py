import os


def gen_code(op_name, head=0):

    f = open("c_source/%s.halide_generated.cpp" % op_name, "r")
    x = f.readlines()
    if head == 1:
        with open("op/generated.h", "w") as fp:
            for line in x[:2363]:
                fp.write(line)

    with open("c_source/%s.cpp" % op_name, "w") as fp:
        headline = "#include \"generated.h\"\n"
        fp.write(headline)
        for line in x[2364:]:
            fp.write(line)
    f.close()


os.system("bash op_build.sh")

gen_code("halide_relu", 1)
gen_code("halide_conv")
gen_code("halide_maxpool")
gen_code("halide_matmul")

os.system("rm c_source/*.halide_generated.cpp")
