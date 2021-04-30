#include <stdio.h>
#include <cmath>

extern "C"
{
    #include "device/cpu/cpu_define.h"
    #include "device/cpu/cpu_node.h"
    #include "device/cpu/cpu_module.h"
    #include "device/cpu/cpu_graph.h"

    #include "api/c_api.h"
    #include "device/device.h"
    #include "graph/tensor.h"
    #include "graph/node.h"
    #include "graph/graph.h"
    #include "graph/subgraph.h"
    #include "executer/executer.h"
    #include "optimizer/split.h"
    #include "module/module.h"
    #include "utility/vector.h"
    #include "utility/log.h"
    #include "utility/sys_port.h"
    #include "defines.h"

    // include op param header file here, locate in src/op/
    #include "operator/prototype/convolution_param.h"
}
#include "HalideBuffer.h"
#include "halide_im2col_conv.h"

int RegisterAutoKernelIm2col_conv(); 
