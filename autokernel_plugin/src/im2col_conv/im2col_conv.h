#include <stdio.h>
#include <cmath>

extern "C"
{
#include "sys_port.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "vector.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "convolution_param.h"

#include "cpu_node_ops.h"
#include "module.h"
}
#include "HalideBuffer.h"
#include "halide_im2col_conv.h"

void RegisterAutoKernelIm2col_conv(); 
