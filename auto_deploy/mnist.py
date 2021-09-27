from graph import Graph
from pass_manager import pass_m
import cv2
import numpy as np
import array

inp_dim = [1, 1, 28, 28]

def gen_main_cpp():
    graph = Graph('mnist', './data/mnist-8.onnx', inp_dim)
    graph = pass_m.run_all_pass(graph)
    graph.infershape()
    graph.gen_main_cpp('c_source/main.cpp')

gen_main_cpp()