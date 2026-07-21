#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Result codes returned by the cudacc C API.
 *
 * Every cudacc API function returns one of these values. Use
 * cudaccGetErrorString to obtain a stable string for logging or diagnostics.
 * Detailed compiler, linker, and tool diagnostics are returned through
 * cudaccOutput::program_log by cudaccCompile.
 */
typedef enum cudaccResult
{
  CUDACC_SUCCESS = 0,
  CUDACC_ERROR_INVALID_INPUT,
  CUDACC_ERROR_INVALID_OPTION,
  CUDACC_ERROR_COMPILATION,
  CUDACC_ERROR_LINKING,
  CUDACC_ERROR_PCH_CREATE,
  CUDACC_ERROR_INTERNAL_ERROR
} cudaccResult;

/**
 * \brief Output and diagnostics produced by cudaccCompile.
 *
 * cudaccCompile initializes this structure before doing any work, so callers
 * may pass an uninitialized cudaccOutput object. output_data is an owned
 * binary buffer and is not NUL-terminated; use output_size to inspect it.
 * program_log is an owned NUL-terminated diagnostic string, and
 * program_log_size is the number of bytes before the trailing NUL.
 *
 * On every cudaccCompile return except when the cudaccOutput pointer itself is
 * NULL, program_log is non-NULL. An empty diagnostic log is represented as
 * program_log_size == 0 and program_log[0] == '\0'. Path-only successful
 * operations return output_data == NULL and output_size == 0. Failed
 * operations return output_data == NULL and put any available diagnostics in
 * program_log.
 *
 * The caller owns neither pointer directly. Release both buffers with
 * cudaccDestroyOutput. Do not modify or free output_data or program_log.
 */
typedef struct cudaccOutput
{
  const char* output_data;
  size_t output_size;
  const char* program_log;
  size_t program_log_size;
} cudaccOutput;

/**
 * \brief Return a static string describing a cudacc result code.
 *
 * The returned pointer is owned by cudacc and remains valid for the lifetime
 * of the process. Unknown result codes return "CUDACC_ERROR_UNKNOWN".
 */
const char* cudaccGetErrorString(cudaccResult result);

/**
 * \brief Compile, create PCHs, or link with an NVRTC-style option array.
 *
 * \param output Output object initialized by cudaccCompile. Passing NULL
 * returns CUDACC_ERROR_INVALID_INPUT and no diagnostic log can be returned.
 * \param options Array of command-line option pointers. The array may be NULL
 * only when num_options is zero.
 * \param num_options Number of entries in options.
 *
 * Exactly one output kind must be specified:
 * - --device-ir: compile one CUDA source to raw LLVM device bitcode returned
 *   in output_data.
 * - --ptx: compile one CUDA source to PTX returned in output_data.
 * - --cubin: produce a linked cubin with nvJitLink returned in output_data.
 * - --fatbin: produce a fatbin with nvFatbin returned in output_data.
 * - --create-device-pch: create a device PCH file. Requires -o <path> and
 *   --pch-source-path=<path> or --pch-source-path <path>.
 * - --create-host-pch: create a host PCH file. Requires -o <path> and
 *   --pch-source-path=<path> or --pch-source-path <path>.
 * - -c or --compile: compile one CUDA source to a host object file at
 *   -o <path> and return the linked device cubin in output_data.
 * - --shared: link host object/archive path inputs into a shared library at
 *   -o <path>.
 *
 * Memory input options always occupy three argv entries: the flag, a
 * NUL-terminated decimal byte size string, and a raw data pointer. The raw
 * data pointer is not interpreted as a string, does not need to be
 * NUL-terminated, and may point to data containing NUL bytes. Supported memory
 * inputs are:
 * - --input-source <size> <data>: one CUDA source buffer. The size is the
 *   exact number of source bytes to compile.
 * - --input-device-ir <size> <data>: raw LLVM device bitcode to link into a
 *   source module before PTX generation.
 * - --input-ptx <size> <data>: PTX input for --cubin or --fatbin.
 * - --input-cubin <size> <data>: cubin input for --cubin or --fatbin.
 * - --input-ltoir <size> <data>: nvJitLink/nvFatbin LTO-IR input.
 *
 * Supported path inputs are:
 * - --input-object=<path> or --input-object <path>: host object input for
 *   --shared.
 * - --input-archive=<path> or --input-archive <path>: host archive/library
 *   input for --shared.
 * - --device-pch=<path> or --device-pch <path>: existing device PCH file.
 * - --host-pch=<path> or --host-pch <path>: existing host PCH file.
 * - --pch-source-path=<path> or --pch-source-path <path>: real source path
 *   recorded by Clang while creating a PCH.
 * - --source-name=<name> or --source-name <name>: logical source name used in
 *   diagnostics for in-memory source inputs. Defaults to input.cu.
 *
 * Supported configuration options are: --cuda-path=<path>,
 * --hostjit-include-path=<path>, --clang-headers-path=<path>,
 * --system-include-path=<path>, -isystem <path>, -isystem<path>,
 * --include-path=<path>, -I <path>, -I<path>, --library-path=<path>,
 * -L <path>, -L<path>, --define-macro=<name>[=<value>],
 * -D <name>[=<value>], -D<name>[=<value>], --gpu-architecture=sm_<NN>,
 * --gpu-architecture=<NN>, --optimization-level=<N>, -O<N>, --debug,
 * --verbose, --trace-includes, --keep-artifacts, --entry-point=<name>,
 * -XClang <arg>, and -XClang=<arg>.
 *
 * Unknown options, malformed options, missing option arguments, non-decimal
 * memory sizes, and invalid numeric option values return
 * CUDACC_ERROR_INVALID_OPTION. Recognized options used in invalid
 * combinations return CUDACC_ERROR_INVALID_INPUT. Device/host compilation
 * failures return CUDACC_ERROR_COMPILATION. Shared-library link failures
 * return CUDACC_ERROR_LINKING. PCH creation failures return
 * CUDACC_ERROR_PCH_CREATE.
 */
cudaccResult cudaccCompile(cudaccOutput* output, const char** options, size_t num_options);

/**
 * \brief Release buffers owned by a cudaccOutput.
 *
 * Passing NULL, a zero-initialized cudaccOutput, or an output object previously
 * initialized by cudaccCompile is accepted. Both pointers are freed, both
 * pointers are set to NULL, both sizes are set to zero, and CUDACC_SUCCESS is
 * returned.
 */
cudaccResult cudaccDestroyOutput(cudaccOutput* output);

#ifdef __cplusplus
}
#endif
