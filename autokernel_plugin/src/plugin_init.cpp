#include <stdio.h>
#include "pool/pool.h"
#include "direct_conv/direct_conv.h"
#include "im2col_conv/im2col_conv.h"
#include "fc/fc.h"
#include "depthwise/depthwise.h"
#include "softmax/softmax.h"
#include "normalize/normalize.h"

extern "C" int autokernel_plugin_init(void)       
{                                      
    /* register halide operator */
    RegisterAutoKernelDepthwise();
    RegisterAutoKernelSoftmax();
    RegisterAutoKernelFc();
    RegisterAutoKernelPool();
    RegisterAutoKernelDirect_conv();
    RegisterAutoKernelIm2col_conv();
    RegisterAutoKernelNormalize();
    printf("AutoKernel plugin inited\n");  
    return 0;                          
}
