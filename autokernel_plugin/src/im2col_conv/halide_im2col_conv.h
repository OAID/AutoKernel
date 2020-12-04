#ifndef HALIDE_____halide_im2col_conv_h
#define HALIDE_____halide_im2col_conv_h
#include <stdint.h>

// Forward declarations of the types used in the interface
// to the Halide pipeline.
//
// For the definitions of these structs, include HalideRuntime.h

// Halide's representation of a multi-dimensional array.
// Halide::Runtime::Buffer is a more user-friendly wrapper
// around this. Its declaration is in HalideBuffer.h
struct halide_buffer_t;

// Metadata describing the arguments to the generated function.
// Used to construct calls to the _argv version of the function.
struct halide_filter_metadata_t;

#ifndef HALIDE_MUST_USE_RESULT
#ifdef __has_attribute
#if __has_attribute(nodiscard)
#define HALIDE_MUST_USE_RESULT [[nodiscard]]
#elif __has_attribute(warn_unused_result)
#define HALIDE_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define HALIDE_MUST_USE_RESULT
#endif
#else
#define HALIDE_MUST_USE_RESULT
#endif
#endif

#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif



#ifdef __cplusplus
extern "C" {
#endif

HALIDE_FUNCTION_ATTRS
int halide_im2col_conv(struct halide_buffer_t *_A__buffer, struct halide_buffer_t *_B__buffer, struct halide_buffer_t *_result_buffer);

HALIDE_FUNCTION_ATTRS
int halide_im2col_conv_argv(void **args);

HALIDE_FUNCTION_ATTRS
const struct halide_filter_metadata_t *halide_im2col_conv_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
