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
 * Detailed compiler, linker, and tool diagnostics are stored on the program
 * handle and can be queried with cudaccGetProgramLogSize and
 * cudaccGetProgramLog.
 */
typedef enum cudaccResult
{
  CUDACC_SUCCESS = 0,
  CUDACC_ERROR_OUT_OF_MEMORY,
  CUDACC_ERROR_PROGRAM_CREATION_FAILURE,
  CUDACC_ERROR_INVALID_INPUT,
  CUDACC_ERROR_INVALID_PROGRAM,
  CUDACC_ERROR_INVALID_OPTION,
  CUDACC_ERROR_COMPILATION,
  CUDACC_ERROR_LINKING,
  CUDACC_ERROR_PCH_CREATE,
  CUDACC_ERROR_INTERNAL_ERROR
} cudaccResult;

/**
 * \brief Selects which Clang compilation mode is used to create a PCH.
 *
 * `CUDACC_PCH_DEVICE` creates a device-side PCH that can later be supplied to
 * cudaccCompileProgramToObject or cudaccCompileProgramToDeviceBitcode with
 * `--device-pch=<path>`. `CUDACC_PCH_HOST` creates a host-side PCH that can
 * later be supplied to cudaccCompileProgramToObject with `--host-pch=<path>`.
 */
typedef enum cudaccPCHKind
{
  CUDACC_PCH_DEVICE = 0,
  CUDACC_PCH_HOST   = 1
} cudaccPCHKind;

/**
 * \brief Opaque cudacc program handle.
 *
 * A program owns the CUDA source string supplied to cudaccCreateProgram and
 * stores diagnostics from the most recent cudacc operation involving that
 * program. Destroy it with cudaccDestroyProgram when no further compilation,
 * PCH creation, linking, or log retrieval is required.
 */
typedef struct cudaccProgram_st* cudaccProgram;

/**
 * \brief Return a static string describing a cudacc result code.
 *
 * The returned pointer is owned by cudacc and remains valid for the lifetime
 * of the process. Unknown result codes return `"CUDACC_ERROR_UNKNOWN"`.
 */
const char* cudaccGetErrorString(cudaccResult result);

/**
 * \brief Create a cudacc program from a CUDA source string.
 *
 * \param prog Output location for the new program handle.
 * \param src NUL-terminated CUDA C++ source string. cudacc copies this string.
 * \param name Optional logical source name used in diagnostics. When NULL or
 * empty, cudacc uses `"input.cu"`.
 *
 * The program does not perform compilation during creation. Compile, link, and
 * PCH functions accept command-line options independently, similar to NVRTC.
 */
cudaccResult cudaccCreateProgram(cudaccProgram* prog, const char* src, const char* name);

/**
 * \brief Destroy a cudacc program.
 *
 * \param prog Address of a program handle previously returned by
 * cudaccCreateProgram. On success, `*prog` is set to NULL. Passing NULL or a
 * pointer to NULL is accepted and returns CUDACC_SUCCESS.
 */
cudaccResult cudaccDestroyProgram(cudaccProgram* prog);

/**
 * \brief Compile a program's device source to LLVM bitcode and write it to a file.
 *
 * \param prog Program handle created by cudaccCreateProgram.
 * \param outputBitcodePath Destination path for the generated LLVM bitcode.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. The array may be NULL
 * when `numOptions` is zero.
 *
 * Supported options:
 * `--cuda-path=<path>`, `--hostjit-include-path=<path>`,
 * `--clang-headers-path=<path>`, `-isystem <path>`, `-isystem<path>`,
 * `--system-include-path=<path>`, `-I<path>`, `--include-path=<path>`, `-L<path>`,
 * `--library-path=<path>`, `--device-bitcode=<path>`,
 * `--device-ltoir=<path>`, `-D<name>[=<value>]`,
 * `--define-macro=<name>[=<value>]`, `--gpu-architecture=sm_<NN>`,
 * `--gpu-architecture=<NN>`, `-O<N>`, `--optimization-level=<N>`,
 * `--debug`, `--verbose`, `--trace-includes`, `--keep-artifacts`,
 * `--entry-point=<name>`, `--device-pch=<path>`, `--host-pch=<path>`,
 * `-XClang <arg>`, and `-XClang=<arg>`.
 *
 * This function uses only file paths for extra LLVM inputs. Source code is the
 * only in-memory input accepted by cudacc.
 */
cudaccResult cudaccCompileProgramToDeviceBitcode(
  cudaccProgram prog, const char* outputBitcodePath, int numOptions, const char* const* options);

/**
 * \brief Compile a program to a host object file and optionally a cubin file.
 *
 * \param prog Program handle created by cudaccCreateProgram.
 * \param outputObjectPath Destination path for the generated host object file.
 * \param outputCubinPath Optional destination path for the linked device cubin.
 * Pass NULL or an empty string when the cubin is not needed.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. The array may be NULL
 * when `numOptions` is zero.
 *
 * Device LLVM bitcode and LTOIR inputs must be supplied with
 * `--device-bitcode=<path>` and `--device-ltoir=<path>`. PCH files are used
 * only when explicit `--device-pch=<path>` or `--host-pch=<path>` options are
 * present; cudacc does not create or cache them implicitly.
 */
cudaccResult cudaccCompileProgramToObject(
  cudaccProgram prog,
  const char* outputObjectPath,
  const char* outputCubinPath,
  int numOptions,
  const char* const* options);

/**
 * \brief Link object files into a shared library.
 *
 * \param prog Program handle used to store diagnostics from the link step.
 * \param numObjectFiles Number of entries in `objectFiles`.
 * \param objectFiles Array of object file paths to link.
 * \param outputLibraryPath Destination path for the linked shared library.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. Link-time options use
 * the same option parser as compile-time options; currently `--cuda-path`,
 * `-L`, `--library-path`, and `--verbose` affect linking.
 */
cudaccResult cudaccLinkToSharedLibrary(
  cudaccProgram prog,
  int numObjectFiles,
  const char* const* objectFiles,
  const char* outputLibraryPath,
  int numOptions,
  const char* const* options);

/**
 * \brief Create a Clang PCH file for a program.
 *
 * \param prog Program handle whose source string is used as the PCH input.
 * \param kind Selects device or host compilation mode.
 * \param pchSourcePath Stable source path to write before invoking Clang.
 * Clang records this path in the PCH, so callers should use a cache-stable
 * location rather than a per-build temporary path.
 * \param pchOutputPath Destination path for the generated PCH file.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. PCH creation uses the
 * same include, macro, architecture, optimization, and `-XClang` options as
 * compilation.
 *
 * cudacc creates exactly the requested PCH file. It does not decide cache
 * locations, check freshness, or enable PCH use for later compilations.
 */
cudaccResult cudaccCreatePCH(
  cudaccProgram prog,
  cudaccPCHKind kind,
  const char* pchSourcePath,
  const char* pchOutputPath,
  int numOptions,
  const char* const* options);

/**
 * \brief Get the byte size of the program diagnostic log.
 *
 * The returned size includes the trailing NUL byte. Warnings and informational
 * messages may be present even when the preceding operation returned
 * CUDACC_SUCCESS.
 */
cudaccResult cudaccGetProgramLogSize(cudaccProgram prog, size_t* logSizeRet);

/**
 * \brief Copy the program diagnostic log into caller-provided storage.
 *
 * The caller must allocate at least the number of bytes returned by
 * cudaccGetProgramLogSize. The copied log is NUL-terminated.
 */
cudaccResult cudaccGetProgramLog(cudaccProgram prog, char* log);

#ifdef __cplusplus
}
#endif
