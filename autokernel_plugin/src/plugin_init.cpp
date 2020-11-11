#include <stdio.h>
#include "direct_conv/direct_conv.h"
#include "im2col_conv/im2col_conv.h"

extern "C" int autokernel_plugin_init(void)       
{                                      
    /* register halide operator */     
    RegisterAutoKernelDirect_conv();
    RegisterAutoKernelIm2col_conv();
    printf("AutoKernel plugin inited\n");  
    return 0;                          
}
