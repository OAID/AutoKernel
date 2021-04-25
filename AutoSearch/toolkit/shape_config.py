
""" Shape configurations for single operator and subgraph evaluation """

matmul_args = [
    [1, 512, 512, 512],
    # [1, 256, 256, 256],
]

matmul_shapes = [
    [[512, 512, 1],[512, 512, 1],[512, 512, 1]],
    # [[256, 256, 1],[256, 256, 1],[256, 256, 1]],
]

shape_dict = {
    'matmul': matmul_shapes,
}

args_dict = {
    'matmul': matmul_args,
}