#include <stdio.h>
#include <math.h>

extern "C"
{
    #include "sys_port.h"
    #include "tengine_errno.h"
    #include "tengine_log.h"
    #include "vector.h"
    #include "tengine_ir.h"
    #include "tengine_op.h"
    #include "../../dev/cpu/cpu_node_ops.h" 

    // include op param header file here, locate in src/op/
    #include "convolution_param.h"
}
                                            
#include "HalideBuffer.h"

// include the c_header file here
#include "halide_depthwise.h"

void RegisterAutoKernelDepthwise();
