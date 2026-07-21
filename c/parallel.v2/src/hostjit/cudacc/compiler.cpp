#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/DiagnosticDriver.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Job.h>
#include <clang/Driver/Tool.h>
#include <clang/Driver/ToolChain.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <cudacc/cudacc.h>
#include <lld/Common/Driver.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/thread.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

// Selective target initialization (native host target plus NVPTX for device)
extern "C" {
void LLVMInitializeX86TargetInfo();
void LLVMInitializeX86Target();
void LLVMInitializeX86TargetMC();
void LLVMInitializeX86AsmPrinter();
void LLVMInitializeX86AsmParser();
void LLVMInitializeNVPTXTargetInfo();
void LLVMInitializeNVPTXTarget();
void LLVMInitializeNVPTXTargetMC();
void LLVMInitializeNVPTXAsmPrinter();
}

#ifdef _WIN32
LLD_HAS_DRIVER(coff)
#else
LLD_HAS_DRIVER(elf)
#endif

#ifdef _WIN32
#  include <llvm/Object/COFFImportFile.h>
#endif

#include <atomic>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <nvFatbin.h>
#include <nvJitLink.h>

namespace cudacc
{
static std::once_flag llvm_init_flag;

static void initialize_llvm()
{
  std::call_once(llvm_init_flag, [] {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

// Embedding clang as a library bypasses the clang driver's
// runWithSufficientStackSpace guard, so the frontend runs on the caller's stack.
// On Windows the default main-thread stack is only 1 MB, which the deep
// (recursive-descent / template-instantiation) frontend overflows on heavier
// kernels such as radix_sort / segmented_reduce; Linux's 8 MB default hides it.
// Run the frontend on a worker thread sized to match clang's own
// DesiredStackSize (8 MB), which is the proven-sufficient value on Linux.
inline constexpr unsigned kFrontendStackSize = 8u << 20;

template <class Fn>
static bool runWithLargeStack(Fn&& fn)
{
  bool result = false;
  llvm::thread worker(std::optional<unsigned>(kFrontendStackSize), [&] {
    result = fn();
  });
  worker.join();
  return result;
}

struct CompilerOptions
{
  std::string cuda_toolkit_path;
  std::string hostjit_include_path;
  std::string clang_headers_path;
  std::string device_pch_path;
  std::string host_pch_path;
  std::string entry_point_name;
  std::vector<std::string> system_include_paths;
  std::vector<std::string> include_paths;
  std::vector<std::string> library_paths;
  std::unordered_map<std::string, std::string> macro_definitions;
  std::vector<std::string> extra_clang_args;
  int sm_version         = 75;
  int optimization_level = 2;
  bool debug             = false;
  bool verbose           = false;
  bool trace_includes    = false;
  bool keep_artifacts    = false;
};

enum class OutputKind
{
  none,
  device_ir,
  ptx,
  cubin,
  fatbin,
  create_device_pch,
  create_host_pch,
  compile,
  shared,
};

struct MemoryInput
{
  std::span<const char> data;
  std::string generated_name;
};

struct CompileRequest
{
  OutputKind output_kind = OutputKind::none;
  CompilerOptions config;
  bool has_source = false;
  std::span<const char> source;
  std::string source_name = "input.cu";
  std::string output_path;
  std::string pch_source_path;
  std::vector<MemoryInput> device_ir_inputs;
  std::vector<MemoryInput> ptx_inputs;
  std::vector<MemoryInput> cubin_inputs;
  std::vector<MemoryInput> ltoir_inputs;
  std::vector<std::string> object_paths;
  std::vector<std::string> archive_paths;
  std::vector<std::string> link_input_paths;
};

struct CompilationResult
{
  bool success = false;
  std::string object_file_path;
  std::vector<char> cubin;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

struct BitcodeResult
{
  bool success = false;
  std::vector<char> bitcode;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

struct PtxResult
{
  bool success = false;
  std::vector<char> ptx;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

struct CubinResult
{
  bool success = false;
  std::vector<char> cubin;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

struct FatbinResult
{
  bool success = false;
  std::vector<char> fatbin;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

struct LinkResult
{
  bool success = false;
  std::string library_path;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

struct DeviceModuleResult
{
  bool success = false;
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

static bool pathExists(const std::filesystem::path& path);

static void addDefaultCudaLibraryPath(CompilerOptions& options)
{
  if (!options.cuda_toolkit_path.empty())
  {
    std::filesystem::path lib64_path = std::filesystem::path(options.cuda_toolkit_path) / "lib64";
    std::filesystem::path lib_path   = std::filesystem::path(options.cuda_toolkit_path) / "lib";

    if (pathExists(lib64_path))
    {
      options.library_paths.push_back(lib64_path.string());
    }
    else if (pathExists(lib_path))
    {
      options.library_paths.push_back(lib_path.string());
    }
  }
}

static void setDefaultOptions(CompilerOptions& options)
{
  if (const char* env = std::getenv("CUDA_PATH"))
  {
    options.cuda_toolkit_path = env;
  }
  else if (const char* env = std::getenv("CUDA_HOME"))
  {
    options.cuda_toolkit_path = env;
  }
#ifdef CUDA_TOOLKIT_PATH
  else
  {
    options.cuda_toolkit_path = CUDA_TOOLKIT_PATH;
  }
#endif

  if (const char* env = std::getenv("HOSTJIT_INCLUDE_PATH"))
  {
    options.hostjit_include_path = env;
  }
#ifdef HOSTJIT_INCLUDE_DIR
  else
  {
    options.hostjit_include_path = HOSTJIT_INCLUDE_DIR;
  }
#endif

  if (const char* env = std::getenv("HOSTJIT_CLANG_PATH"))
  {
    options.clang_headers_path = env;
  }
#ifdef CLANG_HEADERS_DIR
  else
  {
    options.clang_headers_path = CLANG_HEADERS_DIR;
  }
#endif
}

static std::string getHostTargetTriple()
{
  return llvm::sys::getDefaultTargetTriple();
}

static std::string getHostCPUName()
{
  auto cpu = llvm::sys::getHostCPUName();
  if (cpu.empty())
  {
    return "generic";
  }
  return cpu.str();
}

static bool pathExists(const std::filesystem::path& path)
{
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

static std::filesystem::path tempDirectoryPath()
{
  std::error_code ec;
  auto path = std::filesystem::temp_directory_path(ec);
  if (!ec)
  {
    return path;
  }
#ifdef _WIN32
  if (const char* env = std::getenv("TEMP"))
  {
    return env;
  }
  if (const char* env = std::getenv("TMP"))
  {
    return env;
  }
#endif
  if (const char* env = std::getenv("TMPDIR"))
  {
    return env;
  }
  return ".";
}

static bool createDirectories(const std::filesystem::path& path, std::string& diagnostics)
{
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec)
  {
    diagnostics += "Failed to create directory " + path.string() + ": " + ec.message() + "\n";
    return false;
  }
  return true;
}

static bool writeFile(const std::string& path,
                      std::span<const char> data,
                      std::string& diagnostics,
                      const std::string& failure_message)
{
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec);
  if (ec)
  {
    diagnostics += failure_message + ": " + ec.message();
    return false;
  }

  out.write(data.data(), data.size());
  out.flush();
  if (out.has_error())
  {
    ec = out.error();
    diagnostics += failure_message + ": " + ec.message();
    out.clear_error();
    return false;
  }

  return true;
}

static void removeAll(const std::filesystem::path& path)
{
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

template <typename Fn>
static void forEachDirectoryEntry(const std::filesystem::path& dir, Fn&& fn)
{
  std::error_code ec;
  for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
  {
    fn(*it);
  }
}

static bool parseInt(const std::string& value, int& out)
{
  if (value.empty())
  {
    return false;
  }

  int parsed        = 0;
  const char* begin = value.data();
  const char* end   = begin + value.size();
  auto [ptr, ec]    = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end)
  {
    return false;
  }
  out = parsed;
  return true;
}

static bool parseGpuArchitecture(const std::string& value, int& sm)
{
  std::string arch = value;
  if (arch.starts_with("sm_"))
  {
    arch.erase(0, 3);
  }
  return parseInt(arch, sm);
}

static bool parseMacroDefinition(const std::string& value, CompilerOptions& options)
{
  if (value.empty())
  {
    return false;
  }
  auto eq = value.find('=');
  if (eq == std::string::npos)
  {
    options.macro_definitions[value] = "";
  }
  else if (eq == 0)
  {
    return false;
  }
  else
  {
    options.macro_definitions[value.substr(0, eq)] = value.substr(eq + 1);
  }
  return true;
}

static bool parseSize(std::string_view value, size_t& out)
{
  if (value.empty())
  {
    return false;
  }

  size_t parsed     = 0;
  const char* begin = value.data();
  const char* end   = begin + value.size();
  auto [ptr, ec]    = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end)
  {
    return false;
  }
  out = parsed;
  return true;
}

static bool setOutputKind(CompileRequest& request, OutputKind kind, std::string& error)
{
  if (request.output_kind != OutputKind::none)
  {
    error = "Multiple output kind options were specified";
    return false;
  }
  request.output_kind = kind;
  return true;
}

static bool takeRequiredNext(size_t& i,
                             size_t num_options,
                             const char* const* raw_options,
                             std::string_view option_name,
                             const char*& value,
                             std::string& error)
{
  if (++i >= num_options || raw_options[i] == nullptr)
  {
    error = std::string(option_name) + " requires an argument";
    return false;
  }
  value = raw_options[i];
  return true;
}

static bool takeOptionValue(std::string_view option,
                            std::string_view flag,
                            size_t& i,
                            size_t num_options,
                            const char* const* raw_options,
                            std::string& value,
                            std::string& error)
{
  if (option == flag)
  {
    const char* next = nullptr;
    if (!takeRequiredNext(i, num_options, raw_options, flag, next, error))
    {
      return false;
    }
    value = next;
    return true;
  }

  const std::string eq_flag = std::string(flag) + "=";
  if (option.starts_with(eq_flag))
  {
    value = std::string(option.substr(eq_flag.size()));
    return true;
  }

  return false;
}

static bool parseMemoryInput(size_t& i,
                             size_t num_options,
                             const char* const* raw_options,
                             std::string_view flag,
                             std::vector<MemoryInput>& inputs,
                             std::string_view extension,
                             std::string& error)
{
  const char* size_arg = nullptr;
  if (!takeRequiredNext(i, num_options, raw_options, flag, size_arg, error))
  {
    return false;
  }

  size_t size = 0;
  if (!parseSize(size_arg, size) || size == 0)
  {
    error = std::string(flag) + " requires a positive decimal byte size";
    return false;
  }

  const char* data = nullptr;
  if (!takeRequiredNext(i, num_options, raw_options, flag, data, error))
  {
    return false;
  }

  inputs.push_back(MemoryInput{{data, size},
                               "input" + std::to_string(inputs.size()) + std::string(extension)});
  return true;
}

static bool parseSourceInput(size_t& i,
                             size_t num_options,
                             const char* const* raw_options,
                             CompileRequest& request,
                             std::string& error)
{
  if (request.has_source)
  {
    error = "Only one --input-source option may be specified";
    return false;
  }

  const char* size_arg = nullptr;
  if (!takeRequiredNext(i, num_options, raw_options, "--input-source", size_arg, error))
  {
    return false;
  }

  size_t size = 0;
  if (!parseSize(size_arg, size) || size == 0)
  {
    error = "--input-source requires a positive decimal byte size";
    return false;
  }

  const char* data = nullptr;
  if (!takeRequiredNext(i, num_options, raw_options, "--input-source", data, error))
  {
    return false;
  }
  request.has_source = true;
  request.source     = {data, size};
  return true;
}

static bool parseOptions(size_t num_options, const char* const* raw_options, CompileRequest& request, std::string& error)
{
  if (num_options > 0 && raw_options == nullptr)
  {
    error = "Options array is null";
    return false;
  }

  setDefaultOptions(request.config);

  auto value_after_equals = [](std::string_view option, std::string_view prefix) -> std::string {
    return std::string(option.substr(prefix.size()));
  };

  for (size_t i = 0; i < num_options; ++i)
  {
    if (raw_options[i] == nullptr)
    {
      error = "Option string is null";
      return false;
    }

    std::string_view option(raw_options[i]);
    std::string value;
    if (option == "--device-ir")
    {
      if (!setOutputKind(request, OutputKind::device_ir, error))
      {
        return false;
      }
    }
    else if (option == "--ptx")
    {
      if (!setOutputKind(request, OutputKind::ptx, error))
      {
        return false;
      }
    }
    else if (option == "--cubin")
    {
      if (!setOutputKind(request, OutputKind::cubin, error))
      {
        return false;
      }
    }
    else if (option == "--fatbin")
    {
      if (!setOutputKind(request, OutputKind::fatbin, error))
      {
        return false;
      }
    }
    else if (option == "--create-device-pch")
    {
      if (!setOutputKind(request, OutputKind::create_device_pch, error))
      {
        return false;
      }
    }
    else if (option == "--create-host-pch")
    {
      if (!setOutputKind(request, OutputKind::create_host_pch, error))
      {
        return false;
      }
    }
    else if (option == "-c" || option == "--compile")
    {
      if (!setOutputKind(request, OutputKind::compile, error))
      {
        return false;
      }
    }
    else if (option == "--shared")
    {
      if (!setOutputKind(request, OutputKind::shared, error))
      {
        return false;
      }
    }
    else if (option == "-o")
    {
      const char* output_path = nullptr;
      if (!takeRequiredNext(i, num_options, raw_options, "-o", output_path, error))
      {
        return false;
      }
      request.output_path = output_path;
    }
    else if (option == "--input-source")
    {
      if (!parseSourceInput(i, num_options, raw_options, request, error))
      {
        return false;
      }
    }
    else if (option == "--input-device-ir")
    {
      if (!parseMemoryInput(i, num_options, raw_options, option, request.device_ir_inputs, ".bc", error))
      {
        return false;
      }
    }
    else if (option == "--input-ptx")
    {
      if (!parseMemoryInput(i, num_options, raw_options, option, request.ptx_inputs, ".ptx", error))
      {
        return false;
      }
    }
    else if (option == "--input-cubin")
    {
      if (!parseMemoryInput(i, num_options, raw_options, option, request.cubin_inputs, ".cubin", error))
      {
        return false;
      }
    }
    else if (option == "--input-ltoir")
    {
      if (!parseMemoryInput(i, num_options, raw_options, option, request.ltoir_inputs, ".ltoir", error))
      {
        return false;
      }
    }
    else if (takeOptionValue(option, "--input-object", i, num_options, raw_options, value, error))
    {
      request.object_paths.push_back(value);
      request.link_input_paths.push_back(value);
    }
    else if (takeOptionValue(option, "--input-archive", i, num_options, raw_options, value, error))
    {
      request.archive_paths.push_back(value);
      request.link_input_paths.push_back(value);
    }
    else if (takeOptionValue(option, "--device-pch", i, num_options, raw_options, value, error))
    {
      request.config.device_pch_path = value;
    }
    else if (takeOptionValue(option, "--host-pch", i, num_options, raw_options, value, error))
    {
      request.config.host_pch_path = value;
    }
    else if (takeOptionValue(option, "--pch-source-path", i, num_options, raw_options, value, error))
    {
      request.pch_source_path = value;
    }
    else if (takeOptionValue(option, "--source-name", i, num_options, raw_options, value, error))
    {
      request.source_name = value;
    }
    else if (option.starts_with("--cuda-path="))
    {
      request.config.cuda_toolkit_path = value_after_equals(option, "--cuda-path=");
    }
    else if (option.starts_with("--hostjit-include-path="))
    {
      request.config.hostjit_include_path = value_after_equals(option, "--hostjit-include-path=");
    }
    else if (option.starts_with("--clang-headers-path="))
    {
      request.config.clang_headers_path = value_after_equals(option, "--clang-headers-path=");
    }
    else if (option.starts_with("--system-include-path="))
    {
      request.config.system_include_paths.push_back(value_after_equals(option, "--system-include-path="));
    }
    else if (option.starts_with("-isystem") && option.size() > 8)
    {
      request.config.system_include_paths.emplace_back(option.substr(8));
    }
    else if (option == "-isystem")
    {
      const char* include_path = nullptr;
      if (!takeRequiredNext(i, num_options, raw_options, "-isystem", include_path, error))
      {
        return false;
      }
      request.config.system_include_paths.emplace_back(include_path);
    }
    else if (option.starts_with("--include-path="))
    {
      request.config.include_paths.push_back(value_after_equals(option, "--include-path="));
    }
    else if (option.starts_with("-I") && option.size() > 2)
    {
      request.config.include_paths.emplace_back(option.substr(2));
    }
    else if (option == "-I")
    {
      const char* include_path = nullptr;
      if (!takeRequiredNext(i, num_options, raw_options, "-I", include_path, error))
      {
        return false;
      }
      request.config.include_paths.emplace_back(include_path);
    }
    else if (option.starts_with("--library-path="))
    {
      request.config.library_paths.push_back(value_after_equals(option, "--library-path="));
    }
    else if (option.starts_with("-L") && option.size() > 2)
    {
      request.config.library_paths.emplace_back(option.substr(2));
    }
    else if (option == "-L")
    {
      const char* library_path = nullptr;
      if (!takeRequiredNext(i, num_options, raw_options, "-L", library_path, error))
      {
        return false;
      }
      request.config.library_paths.emplace_back(library_path);
    }
    else if (option.starts_with("--define-macro="))
    {
      if (!parseMacroDefinition(value_after_equals(option, "--define-macro="), request.config))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-D") && option.size() > 2)
    {
      if (!parseMacroDefinition(std::string(option.substr(2)), request.config))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option == "-D")
    {
      const char* macro_definition = nullptr;
      if (!takeRequiredNext(i, num_options, raw_options, "-D", macro_definition, error)
          || !parseMacroDefinition(macro_definition, request.config))
      {
        error = "-D requires a macro definition";
        return false;
      }
    }
    else if (option.starts_with("--gpu-architecture="))
    {
      if (!parseGpuArchitecture(value_after_equals(option, "--gpu-architecture="), request.config.sm_version))
      {
        error = "Invalid GPU architecture: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("--optimization-level="))
    {
      if (!parseInt(value_after_equals(option, "--optimization-level="), request.config.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-O") && option.size() > 2)
    {
      if (!parseInt(std::string(option.substr(2)), request.config.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option == "--debug")
    {
      request.config.debug = true;
    }
    else if (option == "--verbose")
    {
      request.config.verbose = true;
    }
    else if (option == "--trace-includes")
    {
      request.config.trace_includes = true;
    }
    else if (option == "--keep-artifacts")
    {
      request.config.keep_artifacts = true;
    }
    else if (option.starts_with("--entry-point="))
    {
      request.config.entry_point_name = value_after_equals(option, "--entry-point=");
    }
    else if (option.starts_with("-XClang="))
    {
      request.config.extra_clang_args.emplace_back(option.substr(8));
    }
    else if (option == "-XClang")
    {
      const char* clang_arg = nullptr;
      if (!takeRequiredNext(i, num_options, raw_options, "-XClang", clang_arg, error))
      {
        return false;
      }
      request.config.extra_clang_args.emplace_back(clang_arg);
    }
    else
    {
      error = "Unknown option: " + std::string(option);
      return false;
    }
  }

  if (request.config.library_paths.empty())
  {
    addDefaultCudaLibraryPath(request.config);
  }

  return true;
}

static bool validateOptions(const CompilerOptions& options, std::string* error_message)
{
  if (options.cuda_toolkit_path.empty())
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path not found. Please pass --cuda-path or set CUDA_PATH/CUDA_HOME.";
    }
    return false;
  }

  if (!pathExists(options.cuda_toolkit_path))
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path does not exist: " + options.cuda_toolkit_path;
    }
    return false;
  }

  std::filesystem::path cuda_h = std::filesystem::path(options.cuda_toolkit_path) / "include" / "cuda.h";
  if (!pathExists(cuda_h))
  {
    if (error_message)
    {
      *error_message = "CUDA headers not found at: " + cuda_h.string();
    }
    return false;
  }

  for (const auto& include_path : options.include_paths)
  {
    if (!pathExists(include_path))
    {
      if (error_message)
      {
        *error_message = "Include path does not exist: " + include_path;
      }
      return false;
    }
  }

  for (const auto& include_path : options.system_include_paths)
  {
    if (!pathExists(include_path))
    {
      if (error_message)
      {
        *error_message = "System include path does not exist: " + include_path;
      }
      return false;
    }
  }

  for (const auto& library_path : options.library_paths)
  {
    if (!pathExists(library_path))
    {
      if (error_message)
      {
        *error_message = "Library path does not exist: " + library_path;
      }
      return false;
    }
  }


  if (!options.device_pch_path.empty() && !pathExists(options.device_pch_path))
  {
    if (error_message)
    {
      *error_message = "Device PCH path does not exist: " + options.device_pch_path;
    }
    return false;
  }

  if (!options.host_pch_path.empty() && !pathExists(options.host_pch_path))
  {
    if (error_message)
    {
      *error_message = "Host PCH path does not exist: " + options.host_pch_path;
    }
    return false;
  }

  if (options.sm_version < 30 || options.sm_version > 150)
  {
    if (error_message)
    {
      *error_message = "Invalid SM version: " + std::to_string(options.sm_version) + " (must be between 30 and 150)";
    }
    return false;
  }

  if (options.optimization_level < 0 || options.optimization_level > 3)
  {
    if (error_message)
    {
      *error_message =
        "Invalid optimization level: " + std::to_string(options.optimization_level) + " (must be between 0 and 3)";
    }
    return false;
  }

  return true;
}

static bool requireNoOutputPath(const CompileRequest& request, std::string& error)
{
  if (!request.output_path.empty())
  {
    error = "-o is invalid for the selected output kind";
    return false;
  }
  return true;
}

static bool requireOutputPath(const CompileRequest& request, std::string& error)
{
  if (request.output_path.empty())
  {
    error = "-o <path> is required for the selected output kind";
    return false;
  }
  return true;
}

static bool requireSource(const CompileRequest& request, std::string& error)
{
  if (!request.has_source)
  {
    error = "--input-source is required for the selected output kind";
    return false;
  }
  return true;
}

static bool rejectSharedInputs(const CompileRequest& request, std::string& error)
{
  if (!request.object_paths.empty() || !request.archive_paths.empty())
  {
    error = "Host object and archive inputs are valid only with --shared";
    return false;
  }
  return true;
}

static bool rejectDeviceMemoryInputs(const CompileRequest& request, std::string& error)
{
  if (!request.device_ir_inputs.empty() || !request.ptx_inputs.empty() || !request.cubin_inputs.empty()
      || !request.ltoir_inputs.empty())
  {
    error = "Device memory inputs are invalid for the selected output kind";
    return false;
  }
  return true;
}

static bool validateCompileRequest(const CompileRequest& request, std::string& error, cudaccResult& result)
{
  result = CUDACC_ERROR_INVALID_INPUT;

  if (request.output_kind == OutputKind::none)
  {
    error = "Exactly one output kind option is required";
    return false;
  }
  if (request.source_name.empty())
  {
    error = "--source-name must be non-empty";
    return false;
  }
  std::string config_error;
  if (!validateOptions(request.config, &config_error))
  {
    error = "Configuration error: " + config_error;
    return false;
  }

  for (const auto& path : request.object_paths)
  {
    if (path.empty())
    {
      error = "--input-object path must be non-empty";
      return false;
    }
  }
  for (const auto& path : request.archive_paths)
  {
    if (path.empty())
    {
      error = "--input-archive path must be non-empty";
      return false;
    }
  }

  if (!request.config.device_pch_path.empty()
      && (request.output_kind == OutputKind::shared || request.output_kind == OutputKind::create_host_pch))
  {
    error = "--device-pch is invalid for the selected output kind";
    return false;
  }
  if (!request.config.host_pch_path.empty()
      && request.output_kind != OutputKind::compile && request.output_kind != OutputKind::create_host_pch)
  {
    error = "--host-pch is invalid for the selected output kind";
    return false;
  }

  switch (request.output_kind)
  {
    case OutputKind::device_ir:
      if (!requireSource(request, error) || !requireNoOutputPath(request, error) || !rejectSharedInputs(request, error))
      {
        return false;
      }
      if (!request.device_ir_inputs.empty() || !request.ptx_inputs.empty() || !request.cubin_inputs.empty()
          || !request.ltoir_inputs.empty())
      {
        error = "Device memory inputs are invalid with --device-ir";
        return false;
      }
      return true;

    case OutputKind::ptx:
      if (!requireSource(request, error) || !requireNoOutputPath(request, error) || !rejectSharedInputs(request, error))
      {
        return false;
      }
      if (!request.ptx_inputs.empty() || !request.cubin_inputs.empty() || !request.ltoir_inputs.empty())
      {
        error = "--input-ptx, --input-cubin, and --input-ltoir are invalid with --ptx";
        return false;
      }
      return true;

    case OutputKind::cubin:
    case OutputKind::fatbin:
      if (!requireNoOutputPath(request, error) || !rejectSharedInputs(request, error))
      {
        return false;
      }
      if (!request.has_source && !request.device_ir_inputs.empty())
      {
        error = "--input-device-ir requires --input-source";
        return false;
      }
      if (!request.has_source && request.ptx_inputs.empty() && request.cubin_inputs.empty() && request.ltoir_inputs.empty())
      {
        error = "At least one source, PTX, cubin, or LTOIR input is required";
        return false;
      }
      return true;

    case OutputKind::create_device_pch:
      if (!requireSource(request, error) || !requireOutputPath(request, error) || !rejectSharedInputs(request, error)
          || !rejectDeviceMemoryInputs(request, error))
      {
        return false;
      }
      if (request.pch_source_path.empty())
      {
        error = "--pch-source-path is required for PCH creation";
        return false;
      }
      if (!request.config.host_pch_path.empty())
      {
        error = "--host-pch is invalid with --create-device-pch";
        return false;
      }
      return true;

    case OutputKind::create_host_pch:
      if (!requireSource(request, error) || !requireOutputPath(request, error) || !rejectSharedInputs(request, error)
          || !rejectDeviceMemoryInputs(request, error))
      {
        return false;
      }
      if (request.pch_source_path.empty())
      {
        error = "--pch-source-path is required for PCH creation";
        return false;
      }
      if (!request.config.device_pch_path.empty())
      {
        error = "--device-pch is invalid with --create-host-pch";
        return false;
      }
      return true;

    case OutputKind::compile:
      if (!requireSource(request, error) || !requireOutputPath(request, error) || !rejectSharedInputs(request, error))
      {
        return false;
      }
      if (!request.ptx_inputs.empty() || !request.cubin_inputs.empty())
      {
        error = "--input-ptx and --input-cubin are invalid with --compile";
        return false;
      }
      return true;

    case OutputKind::shared:
      if (!requireOutputPath(request, error))
      {
        return false;
      }
      if (request.has_source || !request.pch_source_path.empty() || !request.config.device_pch_path.empty()
          || !request.config.host_pch_path.empty() || !request.device_ir_inputs.empty() || !request.ptx_inputs.empty()
          || !request.cubin_inputs.empty() || !request.ltoir_inputs.empty())
      {
        error = "--shared accepts only host object/archive path inputs and link options";
        return false;
      }
      if (request.object_paths.empty() && request.archive_paths.empty())
      {
        error = "--shared requires at least one --input-object or --input-archive";
        return false;
      }
      return true;

    case OutputKind::none:
      break;
  }

  error = "Unhandled output kind";
  return false;
}

static void appendExtraClangArgs(std::vector<std::string>& args, const CompilerOptions& options)
{
  args.insert(args.end(), options.extra_clang_args.begin(), options.extra_clang_args.end());
}

static void appendIncludePaths(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& include_path : options.include_paths)
  {
    args.push_back("-I" + include_path);
  }
}

static void appendMacroDefinitions(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& [macro_name, macro_value] : options.macro_definitions)
  {
    if (macro_value.empty())
    {
      args.push_back("-D" + macro_name);
    }
    else
    {
      args.push_back("-D" + macro_name + "=" + macro_value);
    }
  }
}

// PTX version floor is 7.8. Some generated device code uses features added in
// PTX 7.6 (e.g. `bmsk`), so older versions can fail to assemble even on sm_75/sm_80.
static int ptxVersionForSM(int sm_version)
{
  if (sm_version >= 120)
  {
    return 87;
  }
  if (sm_version >= 100)
  {
    return 85;
  }
  if (sm_version >= 90)
  {
    return 80;
  }
  return 78;
}

static void appendXclangArg(std::vector<std::string>& args, const std::string& arg)
{
  args.push_back("-Xclang");
  args.push_back(arg);
}

static void appendXclangArg(std::vector<std::string>& args, const char* arg)
{
  args.push_back("-Xclang");
  args.push_back(arg);
}

static void appendDriverInternalSystemInclude(std::vector<std::string>& args, const std::string& include_path)
{
  appendXclangArg(args, "-internal-isystem");
  appendXclangArg(args, include_path);
}

static void appendCommonClangDriverArgs(std::vector<std::string>& args,
                                        const std::string& source_file,
                                        const std::string& output_file,
                                        const CompilerOptions& config,
                                        bool is_device)
{
  // Common driver options.
  args.push_back("clang++");
  args.push_back("-x");
  args.push_back("cuda");
  args.push_back(source_file);
  args.push_back("-target");
  args.push_back(getHostTargetTriple());
  args.push_back(is_device ? "--cuda-device-only" : "--cuda-host-only");
  args.push_back("--cuda-gpu-arch=sm_" + std::to_string(config.sm_version));
  args.push_back("--cuda-path=" + config.cuda_toolkit_path);
  args.push_back("-Wno-unknown-cuda-version");
  args.push_back("-fno-exceptions");
  args.push_back("-fno-discard-value-names");
  args.push_back("-nocudainc");
  args.push_back("-nostdinc");
  args.push_back("-nostdinc++");
  args.push_back("-nobuiltininc");
  args.push_back("-resource-dir");
  args.push_back(CLANG_RESOURCE_DIR);
  args.push_back("--offload-new-driver");
  args.push_back("-O" + std::to_string(config.optimization_level));
  args.push_back("-Wno-c++11-narrowing");
  args.push_back("-DNDEBUG");

#ifdef _WIN32
  args.push_back("-fms-compatibility");
#endif

  if (!output_file.empty())
  {
    args.push_back("-o");
    args.push_back(output_file);
  }

  if (config.trace_includes)
  {
    args.push_back("-H");
  }

  // Common cc1 options.
  appendXclangArg(args, "-target-sdk-version=" CUDA_SDK_VERSION);
  appendXclangArg(args, "-fcuda-allow-variadic-functions");
  appendXclangArg(args, "-fdeprecated-macro");

  // Device/host-specific options.
  if (is_device)
  {
    args.push_back("-S");
    args.push_back("-emit-llvm");
    appendXclangArg(args, "-aux-target-cpu");
    appendXclangArg(args, getHostCPUName());
    appendXclangArg(args, "-target-cpu");
    appendXclangArg(args, "sm_" + std::to_string(config.sm_version));
    appendXclangArg(args, "-target-feature");
    appendXclangArg(args, "+ptx" + std::to_string(ptxVersionForSM(config.sm_version)));
    args.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
  }
  else
  {
    args.push_back("-c");
    args.push_back("-fPIC");
    appendXclangArg(args, "-target-cpu");
    appendXclangArg(args, getHostCPUName());
    appendXclangArg(args, "-mrelocation-model");
    appendXclangArg(args, "pic");
    appendXclangArg(args, "-pic-level");
    appendXclangArg(args, "2");
#ifdef _WIN32
    // We do not have access to the windows CRT, so the guard support that
    // threadsafe statics need (_tls_index, _Init_thread_epoch, ...) is
    // unavailable and must be disabled. Generated code IS invoked from
    // multiple threads: first_call_gate (util/first_call_gate.h) serializes
    // the first call into each generated function so its function-local
    // statics initialize race-free despite this flag.
    args.push_back("-fno-threadsafe-statics");
#endif
  }

  // Include paths and other user-specified options.
  appendDriverInternalSystemInclude(args, config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
  appendDriverInternalSystemInclude(
    args, config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
  for (const auto& include_path : config.system_include_paths)
  {
    appendDriverInternalSystemInclude(args, include_path);
  }
  appendDriverInternalSystemInclude(args, config.cuda_toolkit_path + "/include");
  args.push_back("-include");
  args.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");
  appendIncludePaths(args, config);
  appendMacroDefinitions(args, config);
  appendExtraClangArgs(args, config);
}

static bool createInvocationFromDriverArgs(const std::vector<std::string>& driver_arg_strings,
                                           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                                           clang::DiagnosticsEngine& diag_engine,
                                           clang::CompilerInvocation& invocation,
                                           std::vector<std::string>* cc1_arg_strings,
                                           std::string& diagnostics)
{
  std::vector<const char*> driver_args;
  driver_args.reserve(driver_arg_strings.size());
  for (const auto& arg : driver_arg_strings)
  {
    driver_args.push_back(arg.c_str());
  }

  if (driver_args.empty())
  {
    diagnostics += "\nClang driver received no arguments";
    return false;
  }

  clang::driver::Driver driver(
    driver_arg_strings.front(), getHostTargetTriple(), diag_engine, "cudacc clang driver", vfs);
  driver.setCheckInputsExist(false);
  driver.setProbePrecompiled(false);
  driver.setTargetAndMode(clang::driver::ToolChain::getTargetAndModeFromProgramName(driver_arg_strings.front()));
  diag_engine.setSeverity(
    clang::diag::warn_drv_new_cuda_version, clang::diag::Severity::Ignored, clang::SourceLocation());
  diag_engine.setSeverity(
    clang::diag::warn_drv_partially_supported_cuda_version, clang::diag::Severity::Ignored, clang::SourceLocation());

  std::unique_ptr<clang::driver::Compilation> compilation(driver.BuildCompilation(driver_args));
  if (!compilation)
  {
    diagnostics += "\nClang driver failed to build a compilation";
    return false;
  }

  const clang::driver::Command* clang_command = nullptr;
  for (const auto& job : compilation->getJobs())
  {
    if (llvm::StringRef(job.getCreator().getName()) != "clang")
    {
      continue;
    }
    if (clang_command != nullptr)
    {
      diagnostics += "\nClang driver produced multiple clang jobs";
      return false;
    }
    clang_command = &job;
  }

  if (clang_command == nullptr)
  {
    diagnostics += "\nClang driver did not produce a clang job";
    return false;
  }

  const auto& cc1_args = clang_command->getArguments();
  if (cc1_arg_strings)
  {
    cc1_arg_strings->assign(cc1_args.begin(), cc1_args.end());
  }
  return clang::CompilerInvocation::CreateFromArgs(invocation, cc1_args, diag_engine);
}

static void appendClangArgsLog(std::string& diagnostics,
                               const std::string& label,
                               const std::vector<std::string>& args)
{
  diagnostics += label + ": ";
  for (const auto& arg : args)
  {
    diagnostics += arg + " ";
  }
  diagnostics += "\n";
}

static clang::DiagnosticOptions createDiagnosticOptions()
{
  clang::DiagnosticOptions diag_opts;
  diag_opts.ShowColors = false;
  return diag_opts;
}

struct ClangCompilerSetup
{
  std::string diag_output;
  llvm::raw_string_ostream diag_stream;
  clang::DiagnosticOptions diag_opts;
  clang::TextDiagnosticPrinter* diag_printer;
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids;
  clang::DiagnosticsEngine diag_engine;
  clang::CompilerInstance compiler;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs;

  explicit ClangCompilerSetup(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs)
      : diag_stream(diag_output)
      , diag_opts(createDiagnosticOptions())
      , diag_printer(new clang::TextDiagnosticPrinter(diag_stream, diag_opts))
      , diag_ids(new clang::DiagnosticIDs())
      , diag_engine(diag_ids, diag_opts, diag_printer)
      , vfs(fs)
  {}

  clang::CompilerInvocation& invocation()
  {
    return compiler.getInvocation();
  }

  void appendDiagnosticsTo(std::string& diagnostics)
  {
    diag_stream.flush();
    diagnostics += diag_output;
  }
};

static std::unique_ptr<ClangCompilerSetup> createClangCompilerSetup(
  std::vector<std::string> driver_arg_strings,
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
  const std::string& pch_path,
  const CompilerOptions& config,
  const std::string& log_label,
  std::string& diagnostics,
  const std::string& failure_message)
{
  if (!pch_path.empty())
  {
    appendXclangArg(driver_arg_strings, "-include-pch");
    appendXclangArg(driver_arg_strings, pch_path);
  }

  if (config.verbose)
  {
    appendClangArgsLog(diagnostics, log_label + " driver args", driver_arg_strings);
  }

  auto setup = std::make_unique<ClangCompilerSetup>(vfs);
  std::vector<std::string> cc1_arg_strings;
  if (!createInvocationFromDriverArgs(
        driver_arg_strings, setup->vfs, setup->diag_engine, setup->invocation(), &cc1_arg_strings, diagnostics))
  {
    setup->appendDiagnosticsTo(diagnostics);
    diagnostics += failure_message;
    return nullptr;
  }

  if (config.verbose)
  {
    appendClangArgsLog(diagnostics, log_label + " args", cc1_arg_strings);
  }

  setup->compiler.createDiagnostics(setup->diag_engine.getClient(), false);
  setup->compiler.setVirtualFileSystem(setup->vfs);
  setup->compiler.createFileManager();
  return setup;
}

#ifdef _WIN32
// Generate a minimal COFF import library for a given DLL.
// This allows linking without requiring the Windows SDK or MSVC .lib files.
// Symbols can be "name" or "name=dllexport" for aliasing.
static bool generateImportLib(
  const std::string& dll_name,
  const std::vector<std::string>& symbols,
  const std::string& output_path,
  bool data_only = false)
{
  std::vector<llvm::object::COFFShortExport> exports;
  for (const auto& sym : symbols)
  {
    llvm::object::COFFShortExport exp;
    auto eq = sym.find('=');
    if (eq != std::string::npos)
    {
      // "atexit=_crt_atexit" means: linker sees "atexit", DLL exports "_crt_atexit"
      exp.Name       = sym.substr(0, eq); // symbol name the linker resolves
      exp.ImportName = sym.substr(eq + 1); // actual DLL export name
    }
    else
    {
      exp.Name = sym;
    }
    exp.Data = data_only;
    exports.push_back(exp);
  }
  auto err = llvm::object::writeImportLibrary(
    dll_name,
    output_path,
    exports,
    llvm::COFF::IMAGE_FILE_MACHINE_AMD64,
    /*MinGW=*/false);
  if (err)
  {
    llvm::consumeError(std::move(err));
    return false;
  }
  return true;
}

// Find the actual DLL filename for cudart (e.g. "cudart64_13.dll") by
// scanning the CUDA toolkit bin directory.
static std::string findCudartDllName(const std::string& cuda_toolkit_path)
{
  namespace fs = std::filesystem;
  for (const auto& subdir : {"bin/x64", "bin"})
  {
    fs::path dir = fs::path(cuda_toolkit_path) / subdir;
    if (!pathExists(dir))
    {
      continue;
    }
    std::string cudart_name;
    forEachDirectoryEntry(dir, [&](const std::filesystem::directory_entry& entry) {
      auto name = entry.path().filename().string();
      if (cudart_name.empty() && name.starts_with("cudart64_") && name.ends_with(".dll"))
      {
        cudart_name = name;
      }
    });
    if (!cudart_name.empty())
    {
      return cudart_name;
    }
  }
  return "cudart64_12.dll"; // fallback
}
#endif

class CompilerImpl
{
public:
  CompilerImpl() {}

  bool generatePCH(std::span<const char> pch_source,
                   const std::string& pch_source_path,
                   const std::string& pch_output_path,
                   std::vector<std::string> arg_strings,
                   const CompilerOptions& config,
                   const std::string& log_label,
                   std::string& diagnostics)
  {
    static std::atomic<unsigned long> temp_counter{0};
    const std::string temp_suffix =
      ".tmp." + std::to_string(llvm::sys::Process::getProcessId()) + "." + std::to_string(temp_counter++);

    const auto preamble_matches = [&] {
      std::ifstream existing(pch_source_path, std::ios::binary);
      if (!existing)
      {
        return false;
      }
      std::vector<char> contents((std::istreambuf_iterator<char>(existing)), std::istreambuf_iterator<char>());
      return contents.size() == pch_source.size()
          && std::memcmp(contents.data(), pch_source.data(), pch_source.size()) == 0;
    };

    if (!preamble_matches())
    {
      const std::string source_temp_path = pch_source_path + temp_suffix;
      if (!writeFile(source_temp_path, pch_source, diagnostics, "Failed to write PCH preamble to " + source_temp_path))
      {
        return false;
      }

      std::error_code rename_error;
      std::filesystem::rename(source_temp_path, pch_source_path, rename_error);
      if (rename_error)
      {
        std::error_code ignored;
        std::filesystem::remove(source_temp_path, ignored);
        if (!preamble_matches())
        {
          diagnostics += "Failed to move PCH preamble into place: " + rename_error.message();
          return false;
        }
      }
    }

    const std::string output_temp_path = pch_output_path + temp_suffix;
    auto setup = createClangCompilerSetup(
      std::move(arg_strings),
      llvm::vfs::getRealFileSystem(),
      {},
      config,
      log_label,
      diagnostics,
      "\nFailed to create PCH compiler invocation");
    if (!setup)
    {
      return false;
    }

    auto& compiler = setup->compiler;
    compiler.getFrontendOpts().OutputFile = output_temp_path;

    clang::GeneratePCHAction pch_action;
    const bool success = runWithLargeStack([&] {
      return compiler.ExecuteAction(pch_action);
    });

    setup->appendDiagnosticsTo(diagnostics);

    if (!success)
    {
      std::error_code ignored;
      std::filesystem::remove(output_temp_path, ignored);
      return false;
    }

    std::error_code rename_error;
    std::filesystem::rename(output_temp_path, pch_output_path, rename_error);
    if (rename_error)
    {
      std::error_code ignored;
      std::filesystem::remove(output_temp_path, ignored);
      if (!pathExists(pch_output_path))
      {
        diagnostics += "Failed to move PCH into place: " + rename_error.message();
        return false;
      }
    }
    return true;
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  createVFSWithSource(std::span<const char> source_code, llvm::StringRef virtual_path)
  {
    auto mem_fs = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    mem_fs->addFile(
      virtual_path,
      0,
      llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(source_code.data(), source_code.size()), virtual_path));

    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(mem_fs);
    return overlay;
  }

  DeviceModuleResult compileSourceToDeviceModule(std::span<const char> source_code,
                                                 const std::string& source_file,
                                                 const CompilerOptions& config)
  {
    DeviceModuleResult result;
    result.context = std::make_unique<llvm::LLVMContext>();

    std::vector<std::string> driver_arg_strings;
    appendCommonClangDriverArgs(driver_arg_strings, source_file, {}, config, /*is_device=*/true);

    auto setup = createClangCompilerSetup(
      std::move(driver_arg_strings),
      createVFSWithSource(source_code, source_file),
      config.device_pch_path,
      config,
      "Device",
      result.diagnostics,
      "\nFailed to create device compiler invocation");
    if (!setup)
    {
      return result;
    }

    auto& compiler = setup->compiler;

    if (config.trace_includes)
    {
      result.diagnostics += "\n=== Device Header Search Paths ===\n";
      const auto& hso = setup->invocation().getHeaderSearchOpts();
      for (const auto& entry : hso.UserEntries)
      {
        result.diagnostics += "  " + entry.Path + "\n";
      }
      result.diagnostics += "=== End Header Search Paths ===\n\n";
    }

    clang::EmitLLVMOnlyAction emit_llvm_action(result.context.get());
    const bool success = runWithLargeStack([&] {
      return compiler.ExecuteAction(emit_llvm_action);
    });

    if (config.trace_includes && compiler.hasSourceManager())
    {
      result.diagnostics += "\n=== Device Included Files ===\n";
      auto& sm = compiler.getSourceManager();
      for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it)
      {
        result.diagnostics += "  " + it->first.getName().str() + "\n";
      }
      result.diagnostics += "=== End Included Files ===\n\n";
    }

    setup->appendDiagnosticsTo(result.diagnostics);

    if (!success)
    {
      return result;
    }

    result.module = emit_llvm_action.takeModule();
    if (!result.module)
    {
      result.diagnostics += "Failed to get LLVM module";
      return result;
    }

    result.success = true;
    return result;
  }

  bool linkDeviceIRInputs(llvm::Module& mod,
                          llvm::LLVMContext& llvm_context,
                          const std::vector<MemoryInput>& bitcode_inputs,
                          const CompilerOptions& config,
                          std::string& diagnostics)
  {
    for (const auto& input : bitcode_inputs)
    {
      llvm::SMDiagnostic err;
      llvm::MemoryBufferRef buffer_ref(
        llvm::StringRef(input.data.data(), input.data.size()), input.generated_name);
      auto bc_mod = llvm::parseIR(buffer_ref, err, llvm_context);
      if (!bc_mod)
      {
        std::string err_msg;
        llvm::raw_string_ostream err_stream(err_msg);
        err.print("cudacc", err_stream);
        diagnostics += "Failed to parse bitcode: " + input.generated_name + "\n" + err_msg + "\n";
        return false;
      }

      if (llvm::Linker::linkModules(mod, std::move(bc_mod)))
      {
        diagnostics += "Failed to link bitcode: " + input.generated_name + "\n";
        return false;
      }
    }

    if (!bitcode_inputs.empty())
    {
      std::string libdevice_path = config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc";
      llvm::SMDiagnostic err;
      auto libdevice = llvm::parseIRFile(libdevice_path, err, llvm_context);
      if (libdevice)
      {
        llvm::Linker::linkModules(mod, std::move(libdevice), llvm::Linker::LinkOnlyNeeded);
      }
    }

    return true;
  }

  std::unique_ptr<llvm::TargetMachine>
  createTargetMachineForModule(llvm::Module& mod, const CompilerOptions& config, std::string& diagnostics)
  {
    std::string err_str;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(mod.getTargetTriple(), err_str);
    if (!target)
    {
      diagnostics += "Failed to lookup target: " + err_str + "\n";
      return nullptr;
    }

    llvm::TargetOptions opt;
    std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      mod.getTargetTriple(),
      "sm_" + std::to_string(config.sm_version),
      "+ptx" + std::to_string(ptxVersionForSM(config.sm_version)),
      opt,
      llvm::Reloc::PIC_));
    if (!tm)
    {
      diagnostics += "Failed to create target machine\n";
      return nullptr;
    }

    mod.setDataLayout(tm->createDataLayout());
    return tm;
  }

  void optimizeDeviceModuleForEntryPoint(
    llvm::Module& mod, llvm::TargetMachine& tm, const CompilerOptions& config, bool force_optimization)
  {
    if (!force_optimization && config.entry_point_name.empty())
    {
      return;
    }

    if (!config.entry_point_name.empty())
    {
      for (auto& F : mod)
      {
        if (!F.isDeclaration() && F.getLinkage() == llvm::GlobalValue::ExternalLinkage
            && F.getName() != config.entry_point_name && F.getCallingConv() != llvm::CallingConv::PTX_Kernel)
        {
          F.setLinkage(llvm::GlobalValue::InternalLinkage);
          F.removeFnAttr(llvm::Attribute::NoInline);
          F.removeFnAttr(llvm::Attribute::OptimizeNone);
          F.addFnAttr(llvm::Attribute::AlwaysInline);
        }
      }
    }

    llvm::OptimizationLevel opt_level;
    switch (config.optimization_level)
    {
      case 0:
        opt_level = llvm::OptimizationLevel::O0;
        break;
      case 1:
        opt_level = llvm::OptimizationLevel::O1;
        break;
      case 3:
        opt_level = llvm::OptimizationLevel::O3;
        break;
      default:
        opt_level = llvm::OptimizationLevel::O2;
        break;
    }

    static const bool unroll_tuned = [] {
      auto& opts   = llvm::cl::getRegisteredOptions();
      auto set_opt = [&](llvm::StringRef name, llvm::StringRef value) {
        auto it = opts.find(name);
        if (it != opts.end())
        {
          it->second->addOccurrence(0, name, value);
        }
      };
      set_opt("unroll-threshold", "4000");
      set_opt("unroll-full-max-count", "1024");
      set_opt("unroll-max-upperbound", "1024");
      return true;
    }();
    (void) unroll_tuned;

    auto run_pipeline = [&](llvm::Module& module) {
      llvm::LoopAnalysisManager LAM;
      llvm::FunctionAnalysisManager FAM;
      llvm::CGSCCAnalysisManager CGAM;
      llvm::ModuleAnalysisManager MAM;

      llvm::PassBuilder PB(&tm);
      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      auto MPM = PB.buildPerModuleDefaultPipeline(opt_level);
      MPM.run(module, MAM);
    };

    run_pipeline(mod);
    run_pipeline(mod);
  }

  bool writeDeviceIRToMemory(llvm::Module& mod, std::vector<char>& bitcode, std::string& diagnostics)
  {
    llvm::SmallVector<char, 0> buffer;
    llvm::raw_svector_ostream os(buffer);
    llvm::WriteBitcodeToFile(mod, os);
    bitcode.assign(buffer.begin(), buffer.end());
    if (bitcode.empty())
    {
      diagnostics += "Failed to write device bitcode to memory";
      return false;
    }
    return true;
  }

  bool emitPTXToMemory(llvm::Module& mod,
                       const CompilerOptions& config,
                       bool force_optimization,
                       std::vector<char>& ptx,
                       std::string& diagnostics)
  {
    auto tm = createTargetMachineForModule(mod, config, diagnostics);
    if (!tm)
    {
      return false;
    }

    optimizeDeviceModuleForEntryPoint(mod, *tm, config, force_optimization);

    llvm::SmallVector<char, 0> buffer;
    llvm::raw_svector_ostream dest(buffer);
    llvm::legacy::PassManager pass;
    if (tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::AssemblyFile))
    {
      diagnostics += "Target machine cannot emit PTX\n";
      return false;
    }
    pass.run(mod);
    ptx.assign(buffer.begin(), buffer.end());
    if (ptx.empty())
    {
      diagnostics += "Failed to emit PTX\n";
      return false;
    }

    if (const char* dump_dir = std::getenv("CUDACC_DUMP_DIR"))
    {
      std::error_code dec;
      std::filesystem::create_directories(dump_dir, dec);
      const std::string base = config.entry_point_name.empty() ? std::string("kernel") : config.entry_point_name;
      const std::string stem = (std::filesystem::path(dump_dir) / base).string();
      llvm::raw_fd_ostream ll_os(stem + ".opt.ll", dec);
      if (!dec)
      {
        mod.print(ll_os, nullptr);
      }
      writeFile(stem + ".ptx", std::span<const char>{ptx.data(), ptx.size()}, diagnostics, "Failed to dump PTX");
      llvm::errs() << "[hostjit] dumped " << stem << ".opt.ll and " << stem << ".ptx\n";
    }

    return true;
  }

  BitcodeResult compileToDeviceBitcode(std::span<const char> source_code,
                                       const std::string& input_name,
                                       const CompilerOptions& config)
  {
    BitcodeResult result;

    std::string temp_dir =
      (tempDirectoryPath() / ("cudacc_bc_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }
    auto cleanup_temp_dir = llvm::scope_exit([&] {
      removeAll(temp_dir);
    });

    const std::string input_file  = input_name.empty() ? std::string("input.cu") : input_name;
    const std::string source_file = (std::filesystem::path(temp_dir) / input_file).string();

    auto module_result = compileSourceToDeviceModule(source_code, source_file, config);
    result.diagnostics += module_result.diagnostics;
    if (!module_result)
    {
      return result;
    }

    if (!writeDeviceIRToMemory(*module_result.module, result.bitcode, result.diagnostics))
    {
      return result;
    }

    if (config.keep_artifacts)
    {
      cleanup_temp_dir.release();
    }
    result.success = true;
    return result;
  }

  PtxResult compileToPTX(std::span<const char> source_code,
                         const std::string& input_name,
                         const std::vector<MemoryInput>& device_ir_inputs,
                         const CompilerOptions& config)
  {
    PtxResult result;

    std::string temp_dir =
      (tempDirectoryPath() / ("cudacc_ptx_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }
    auto cleanup_temp_dir = llvm::scope_exit([&] {
      removeAll(temp_dir);
    });

    const std::string input_file  = input_name.empty() ? std::string("input.cu") : input_name;
    const std::string source_file = (std::filesystem::path(temp_dir) / input_file).string();

    auto module_result = compileSourceToDeviceModule(source_code, source_file, config);
    result.diagnostics += module_result.diagnostics;
    if (!module_result)
    {
      return result;
    }

    if (!linkDeviceIRInputs(*module_result.module, *module_result.context, device_ir_inputs, config, result.diagnostics))
    {
      return result;
    }

    if (!emitPTXToMemory(*module_result.module, config, !device_ir_inputs.empty(), result.ptx, result.diagnostics))
    {
      return result;
    }

    if (config.keep_artifacts)
    {
      cleanup_temp_dir.release();
    }
    result.success = true;
    return result;
  }

  void appendNvJitLinkErrorLog(nvJitLinkHandle jitlink_handle, std::string& diagnostics)
  {
    size_t log_size = 0;
    nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
    if (log_size > 1)
    {
      std::string log(log_size, '\0');
      nvJitLinkGetErrorLog(jitlink_handle, log.data());
      if (!log.empty() && log.back() == '\0')
      {
        log.pop_back();
      }
      diagnostics += "\n" + log;
    }
  }

  CubinResult linkToCubin(std::span<const char> generated_ptx,
                          const std::vector<MemoryInput>& ptx_inputs,
                          const std::vector<MemoryInput>& cubin_inputs,
                          const std::vector<MemoryInput>& ltoir_inputs,
                          const CompilerOptions& config)
  {
    CubinResult result;

    std::string arch_opt  = "-arch=sm_" + std::to_string(config.sm_version);
    std::string opt_level = "-O" + std::to_string(config.optimization_level >= 1 ? 3 : 0);
    std::vector<std::string> jitlink_option_strs{arch_opt, opt_level};
    if (!ltoir_inputs.empty())
    {
      jitlink_option_strs.emplace_back("-lto");
    }

    std::vector<const char*> jitlink_options;
    jitlink_options.reserve(jitlink_option_strs.size());
    for (const auto& option : jitlink_option_strs)
    {
      jitlink_options.push_back(option.c_str());
    }

    nvJitLinkHandle jitlink_handle = nullptr;
    nvJitLinkResult jlr =
      nvJitLinkCreate(&jitlink_handle, static_cast<uint32_t>(jitlink_options.size()), jitlink_options.data());
    if (jlr != NVJITLINK_SUCCESS)
    {
      result.diagnostics += "\nnvJitLinkCreate failed (error " + std::to_string(static_cast<int>(jlr)) + ")";
      result.diagnostics += "\nnvJitLink options:";
      for (const auto& option : jitlink_option_strs)
      {
        result.diagnostics += " " + option;
      }
      return result;
    }
    auto destroy_jitlink_handle = llvm::scope_exit([&] {
      nvJitLinkDestroy(&jitlink_handle);
    });

    std::vector<std::vector<char>> ptx_storage;
    ptx_storage.reserve((generated_ptx.empty() ? 0 : 1) + ptx_inputs.size());

    auto add_ptx = [&](std::span<const char> ptx, const std::string& name) {
      ptx_storage.emplace_back(ptx.begin(), ptx.end());
      if (ptx_storage.back().empty() || ptx_storage.back().back() != '\0')
      {
        ptx_storage.back().push_back('\0');
      }
      jlr = nvJitLinkAddData(
        jitlink_handle, NVJITLINK_INPUT_PTX, ptx_storage.back().data(), ptx_storage.back().size(), name.c_str());
      if (jlr != NVJITLINK_SUCCESS)
      {
        appendNvJitLinkErrorLog(jitlink_handle, result.diagnostics);
        result.diagnostics += "\nnvJitLinkAddData(PTX) failed for " + name;
        return false;
      }
      return true;
    };

    if (!generated_ptx.empty() && !add_ptx(generated_ptx, "source.ptx"))
    {
      return result;
    }
    for (const auto& input : ptx_inputs)
    {
      if (!add_ptx(input.data, input.generated_name))
      {
        return result;
      }
    }
    for (const auto& input : cubin_inputs)
    {
      jlr = nvJitLinkAddData(
        jitlink_handle, NVJITLINK_INPUT_CUBIN, input.data.data(), input.data.size(), input.generated_name.c_str());
      if (jlr != NVJITLINK_SUCCESS)
      {
        appendNvJitLinkErrorLog(jitlink_handle, result.diagnostics);
        result.diagnostics += "\nnvJitLinkAddData(CUBIN) failed for " + input.generated_name;
        return result;
      }
    }
    for (const auto& input : ltoir_inputs)
    {
      jlr = nvJitLinkAddData(
        jitlink_handle, NVJITLINK_INPUT_LTOIR, input.data.data(), input.data.size(), input.generated_name.c_str());
      if (jlr != NVJITLINK_SUCCESS)
      {
        appendNvJitLinkErrorLog(jitlink_handle, result.diagnostics);
        result.diagnostics += "\nnvJitLinkAddData(LTOIR) failed for " + input.generated_name;
        return result;
      }
    }

    jlr = nvJitLinkComplete(jitlink_handle);
    if (jlr != NVJITLINK_SUCCESS)
    {
      appendNvJitLinkErrorLog(jitlink_handle, result.diagnostics);
      result.diagnostics += "\nnvJitLinkComplete failed";
      return result;
    }

    size_t cubin_size = 0;
    jlr               = nvJitLinkGetLinkedCubinSize(jitlink_handle, &cubin_size);
    if (jlr != NVJITLINK_SUCCESS || cubin_size == 0)
    {
      result.diagnostics += "\nnvJitLinkGetLinkedCubinSize failed";
      return result;
    }

    result.cubin.resize(cubin_size);
    jlr = nvJitLinkGetLinkedCubin(jitlink_handle, result.cubin.data());
    if (jlr != NVJITLINK_SUCCESS)
    {
      result.diagnostics += "\nnvJitLinkGetLinkedCubin failed";
      return result;
    }

    result.success = true;
    return result;
  }

  FatbinResult createFatbin(std::span<const char> generated_ptx,
                            std::span<const char> generated_cubin,
                            const std::vector<MemoryInput>& ptx_inputs,
                            const std::vector<MemoryInput>& cubin_inputs,
                            const std::vector<MemoryInput>& ltoir_inputs,
                            const CompilerOptions& config)
  {
    FatbinResult result;

    const char* fatbin_options[] = {"-64", "-cuda"};
    nvFatbinHandle fatbin_handle = nullptr;
    nvFatbinResult fbr           = nvFatbinCreate(&fatbin_handle, fatbin_options, 2);
    if (fbr != NVFATBIN_SUCCESS)
    {
      result.diagnostics += std::string("\nnvFatbinCreate failed: ") + nvFatbinGetErrorString(fbr);
      return result;
    }
    auto destroy_fatbin_handle = llvm::scope_exit([&] {
      nvFatbinDestroy(&fatbin_handle);
    });

    const std::string arch = std::to_string(config.sm_version);

    if (!generated_cubin.empty())
    {
      fbr = nvFatbinAddCubin(fatbin_handle, generated_cubin.data(), generated_cubin.size(), arch.c_str(), "source.cubin");
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddCubin failed: ") + nvFatbinGetErrorString(fbr);
        return result;
      }
    }

    std::vector<std::vector<char>> ptx_storage;
    ptx_storage.reserve((generated_ptx.empty() ? 0 : 1) + ptx_inputs.size());
    auto add_ptx = [&](std::span<const char> ptx, const std::string& name) {
      ptx_storage.emplace_back(ptx.begin(), ptx.end());
      if (ptx_storage.back().empty() || ptx_storage.back().back() != '\0')
      {
        ptx_storage.back().push_back('\0');
      }
      fbr = nvFatbinAddPTX(
        fatbin_handle, ptx_storage.back().data(), ptx_storage.back().size(), arch.c_str(), name.c_str(), nullptr);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddPTX failed: ") + nvFatbinGetErrorString(fbr);
        return false;
      }
      return true;
    };

    if (!generated_ptx.empty() && !add_ptx(generated_ptx, "source.ptx"))
    {
      return result;
    }
    for (const auto& input : ptx_inputs)
    {
      if (!add_ptx(input.data, input.generated_name))
      {
        return result;
      }
    }
    for (const auto& input : cubin_inputs)
    {
      fbr = nvFatbinAddCubin(
        fatbin_handle, input.data.data(), input.data.size(), arch.c_str(), input.generated_name.c_str());
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddCubin failed: ") + nvFatbinGetErrorString(fbr);
        return result;
      }
    }
    for (const auto& input : ltoir_inputs)
    {
      fbr = nvFatbinAddLTOIR(
        fatbin_handle, input.data.data(), input.data.size(), arch.c_str(), input.generated_name.c_str(), nullptr);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddLTOIR failed: ") + nvFatbinGetErrorString(fbr);
        return result;
      }
    }

    size_t fatbin_size = 0;
    fbr                = nvFatbinSize(fatbin_handle, &fatbin_size);
    if (fbr != NVFATBIN_SUCCESS || fatbin_size == 0)
    {
      result.diagnostics += std::string("\nnvFatbinSize failed: ") + nvFatbinGetErrorString(fbr);
      return result;
    }

    result.fatbin.resize(fatbin_size);
    fbr = nvFatbinGet(fatbin_handle, result.fatbin.data());
    if (fbr != NVFATBIN_SUCCESS)
    {
      result.diagnostics += std::string("\nnvFatbinGet failed: ") + nvFatbinGetErrorString(fbr);
      return result;
    }

    result.success = true;
    return result;
  }

  bool compileHostCode(std::span<const char> source_code,
                       const std::string& input_file,
                       const std::string& fatbin_path,
                       const std::string& output_obj,
                       const CompilerOptions& config,
                       std::string& diagnostics)
  {
    std::string temp_dir    = std::filesystem::path(output_obj).parent_path().string();
    std::string source_file = (std::filesystem::path(temp_dir) / ("host_" + input_file)).string();

    std::vector<std::string> driver_arg_strings;
    appendCommonClangDriverArgs(driver_arg_strings, source_file, output_obj, config, /*is_device=*/false);
    appendXclangArg(driver_arg_strings, "-fcuda-include-gpubinary");
    appendXclangArg(driver_arg_strings, fatbin_path);

    auto setup = createClangCompilerSetup(
      std::move(driver_arg_strings),
      createVFSWithSource(source_code, source_file),
      config.host_pch_path,
      config,
      "Host",
      diagnostics,
      "\nFailed to create host compiler invocation");
    if (!setup)
    {
      return false;
    }

    auto& compiler = setup->compiler;
    compiler.getFrontendOpts().OutputFile = output_obj;

    if (config.trace_includes)
    {
      diagnostics += "\n=== Host Header Search Paths ===\n";
      const auto& hso = setup->invocation().getHeaderSearchOpts();
      for (const auto& entry : hso.UserEntries)
      {
        diagnostics += "  " + entry.Path + "\n";
      }
      diagnostics += "=== End Header Search Paths ===\n\n";
    }

    clang::EmitObjAction emit_action;
    const bool success = runWithLargeStack([&] {
      return compiler.ExecuteAction(emit_action);
    });

    if (config.trace_includes && compiler.hasSourceManager())
    {
      diagnostics += "\n=== Host Included Files ===\n";
      auto& sm = compiler.getSourceManager();
      for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it)
      {
        diagnostics += "  " + it->first.getName().str() + "\n";
      }
      diagnostics += "=== End Included Files ===\n\n";
    }

    setup->appendDiagnosticsTo(diagnostics);

    return success;
  }

  CompilationResult compileToObject(std::span<const char> source_code,
                                    const std::string& input_name,
                                    const std::string& output_path,
                                    const std::vector<MemoryInput>& device_ir_inputs,
                                    const std::vector<MemoryInput>& ltoir_inputs,
                                    const CompilerOptions& config)
  {
    CompilationResult result;
    result.object_file_path = output_path;

    std::string temp_dir =
      (tempDirectoryPath() / ("cudacc_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }
    auto cleanup_temp_dir = llvm::scope_exit([&] {
      removeAll(temp_dir);
    });

    const std::string input_file  = input_name.empty() ? std::string("input.cu") : input_name;
    const std::string fatbin_file = (std::filesystem::path(temp_dir) / "device.fatbin").string();

    if (config.verbose)
    {
      result.diagnostics += "=== Device compilation ===\n";
    }

    auto ptx_result = compileToPTX(source_code, input_file, device_ir_inputs, config);
    result.diagnostics += ptx_result.diagnostics;
    if (!ptx_result)
    {
      result.diagnostics += "\nDevice compilation failed";
      return result;
    }

    if (config.verbose)
    {
      result.diagnostics += "\n=== nvJitLink + fatbinary ===\n";
    }

    auto cubin_result = linkToCubin(
      std::span<const char>{ptx_result.ptx.data(), ptx_result.ptx.size()}, {}, {}, ltoir_inputs, config);
    result.diagnostics += cubin_result.diagnostics;
    if (!cubin_result)
    {
      return result;
    }

    auto fatbin_result = createFatbin(
      std::span<const char>{ptx_result.ptx.data(), ptx_result.ptx.size()},
      std::span<const char>{cubin_result.cubin.data(), cubin_result.cubin.size()},
      {},
      {},
      {},
      config);
    result.diagnostics += fatbin_result.diagnostics;
    if (!fatbin_result)
    {
      return result;
    }

    if (!writeFile(
          fatbin_file,
          std::span<const char>{fatbin_result.fatbin.data(), fatbin_result.fatbin.size()},
          result.diagnostics,
          "\nFailed to write fatbin file"))
    {
      return result;
    }

    if (config.verbose)
    {
      result.diagnostics += "\n=== Host compilation ===\n";
    }

    if (!compileHostCode(source_code, input_file, fatbin_file, output_path, config, result.diagnostics))
    {
      result.diagnostics += "\nHost compilation failed";
      return result;
    }

    if (config.keep_artifacts)
    {
      cleanup_temp_dir.release();
    }
    result.cubin   = std::move(cubin_result.cubin);
    result.success = true;
    return result;
  }

  bool createPCH(std::span<const char> source_code,
                 bool is_device,
                 const std::string& pch_source_path,
                 const std::string& pch_output_path,
                 const CompilerOptions& config,
                 std::string& diagnostics)
  {
    std::vector<std::string> arg_strings;
    appendCommonClangDriverArgs(arg_strings, pch_source_path, pch_output_path, config, is_device);
    return generatePCH(
      source_code,
      pch_source_path,
      pch_output_path,
      std::move(arg_strings),
      config,
      is_device ? "Device PCH" : "Host PCH",
      diagnostics);
  }

  LinkResult linkToSharedLibrary(
    const std::vector<std::string>& input_files, const std::string& output_path, const CompilerOptions& config)
  {
    LinkResult result;
    result.library_path = output_path;

    if (input_files.empty())
    {
      result.diagnostics = "No object files provided";
      return result;
    }

    std::vector<std::string> arg_strings;

#ifdef _WIN32
    arg_strings.push_back("lld-link");
    arg_strings.push_back("/DLL");
    arg_strings.push_back("/NOENTRY");
    arg_strings.push_back("/NODEFAULTLIB");
    arg_strings.push_back("/OUT:" + output_path);

    std::string implib_dir = std::filesystem::path(output_path).parent_path().string();

    std::string cudart_dll = findCudartDllName(config.cuda_toolkit_path);
    generateImportLib(
      cudart_dll,
      {"cudaMalloc",
       "cudaFree",
       "cudaMemcpy",
       "cudaMemcpyAsync",
       "cudaMemset",
       "cudaMemsetAsync",
       "cudaDeviceSynchronize",
       "cudaFuncSetAttribute",
       "cudaGetDevice",
       "cudaGetDeviceProperties",
       "cudaGetLastError",
       "cudaPeekAtLastError",
       "cudaGetErrorString",
       "cudaStreamCreate",
       "cudaStreamDestroy",
       "cudaStreamSynchronize",
       "cudaEventCreate",
       "cudaEventDestroy",
       "cudaEventRecord",
       "cudaEventSynchronize",
       "cudaEventElapsedTime",
       "cudaMallocAsync",
       "cudaFreeAsync",
       "cudaDeviceGetAttribute",
       "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
       "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
       "cudaFuncGetAttributes",
       "cudaLaunchKernel",
       "cudaLaunchKernelExC",
       "__cudaRegisterFatBinary",
       "__cudaRegisterFatBinaryEnd",
       "__cudaUnregisterFatBinary",
       "__cudaRegisterFunction",
       "__cudaRegisterVar",
       "__cudaPushCallConfiguration",
       "__cudaPopCallConfiguration"},
      implib_dir + "/cudart.lib");

    generateImportLib(
      "ucrtbase.dll",
      {"malloc",
       "free",
       "calloc",
       "realloc",
       "_callnewh",
       "_errno",
       "abort",
       "exit",
       "_exit",
       "_register_onexit_function",
       "_crt_atexit",
       "_initterm",
       "_initterm_e",
       "memcpy",
       "memset",
       "memmove",
       "memcmp",
       "strlen",
       "strcmp",
       "strncmp",
       "_initialize_onexit_table",
       "_execute_onexit_table",
       "_register_thread_local_exe_atexit_callback"},
      implib_dir + "/ucrt.lib");

    generateImportLib(
      "vcruntime140.dll",
      {"__std_exception_copy",
       "__std_exception_destroy",
       "__CxxFrameHandler3",
       "_CxxThrowException",
       "memcpy",
       "memset",
       "memmove",
       "memcmp",
       "__std_type_info_destroy_list",
       "_purecall"},
      implib_dir + "/vcruntime.lib");

    generateImportLib(
      "kernel32.dll",
      {"InitializeCriticalSection",
       "EnterCriticalSection",
       "LeaveCriticalSection",
       "DeleteCriticalSection",
       "InitOnceExecuteOnce",
       "LoadLibraryExA",
       "LoadLibraryExW",
       "GetProcAddress",
       "FreeLibrary",
       "GetModuleHandleA",
       "GetLastError",
       "SetLastError",
       "GetCurrentProcess",
       "GetCurrentThread",
       "GetCurrentThreadId",
       "VirtualProtect",
       "FlushInstructionCache",
       "QueryPerformanceCounter",
       "QueryPerformanceFrequency"},
      implib_dir + "/kernel32.lib");

    arg_strings.push_back("/LIBPATH:" + implib_dir);

    for (const auto& input_file : input_files)
    {
      arg_strings.push_back(input_file);
    }

    arg_strings.push_back("cudart.lib");
    arg_strings.push_back("ucrt.lib");
    arg_strings.push_back("vcruntime.lib");
    arg_strings.push_back("kernel32.lib");
#else
    arg_strings.push_back("ld.lld");
    arg_strings.push_back("-shared");
    arg_strings.push_back("--build-id");
    arg_strings.push_back("--eh-frame-hdr");
    arg_strings.push_back("--allow-shlib-undefined");
    arg_strings.push_back("-o");
    arg_strings.push_back(output_path);

    for (const auto& lib_path : config.library_paths)
    {
      arg_strings.push_back("-L" + lib_path);
      arg_strings.push_back("-rpath");
      arg_strings.push_back(lib_path);
    }

    for (const auto& input_file : input_files)
    {
      arg_strings.push_back(input_file);
    }

    {
      bool found_cudart = false;
      for (const auto& lib_path : config.library_paths)
      {
        if (!pathExists(lib_path))
        {
          continue;
        }
        forEachDirectoryEntry(lib_path, [&](const std::filesystem::directory_entry& entry) {
          auto fname = entry.path().filename().string();
          if (!found_cudart && fname.starts_with("libcudart.so"))
          {
            arg_strings.push_back(entry.path().string());
            found_cudart = true;
          }
        });
        if (found_cudart)
        {
          break;
        }
      }
      if (!found_cudart)
      {
        arg_strings.push_back("-lcudart");
      }
    }
#endif

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    std::string stdout_str, stderr_str;
    llvm::raw_string_ostream stdout_os(stdout_str);
    llvm::raw_string_ostream stderr_os(stderr_str);

#ifdef _WIN32
    bool link_success = lld::coff::link(args, stdout_os, stderr_os, false, false);
#else
    bool link_success = lld::elf::link(args, stdout_os, stderr_os, false, false);
#endif

    stdout_os.flush();
    stderr_os.flush();

    if (!stdout_str.empty())
    {
      result.diagnostics += stdout_str;
    }
    if (!stderr_str.empty())
    {
      result.diagnostics += stderr_str;
    }

    if (!link_success)
    {
      result.diagnostics += "\nLinking failed";
      return result;
    }

    result.success = true;
    return result;
  }
};
} // namespace cudacc

namespace
{
void initOutput(cudaccOutput& output)
{
  output.output_data      = nullptr;
  output.output_size      = 0;
  output.program_log      = nullptr;
  output.program_log_size = 0;
}

void freeOutput(cudaccOutput& output)
{
  delete[] const_cast<char*>(output.output_data);
  delete[] const_cast<char*>(output.program_log);
  initOutput(output);
}

void setOutputLog(cudaccOutput& output, std::string_view log)
{
  delete[] const_cast<char*>(output.program_log);
  auto* data = new char[log.size() + 1];
  if (!log.empty())
  {
    std::memcpy(data, log.data(), log.size());
  }
  data[log.size()]       = '\0';
  output.program_log      = data;
  output.program_log_size = log.size();
}

void setOutputData(cudaccOutput& output, std::span<const char> data)
{
  delete[] const_cast<char*>(output.output_data);
  output.output_data = nullptr;
  output.output_size = 0;
  if (data.empty())
  {
    return;
  }

  auto* buffer = new char[data.size()];
  std::memcpy(buffer, data.data(), data.size());
  output.output_data = buffer;
  output.output_size = data.size();
}

cudaccResult finishCompile(cudaccOutput& output, cudaccResult result, std::string_view diagnostics)
{
  if (result != CUDACC_SUCCESS)
  {
    delete[] const_cast<char*>(output.output_data);
    output.output_data = nullptr;
    output.output_size = 0;
  }
  setOutputLog(output, diagnostics);
  return result;
}
} // anonymous namespace

extern "C" const char* cudaccGetErrorString(cudaccResult result)
{
  switch (result)
  {
    case CUDACC_SUCCESS:
      return "CUDACC_SUCCESS";
    case CUDACC_ERROR_INVALID_INPUT:
      return "CUDACC_ERROR_INVALID_INPUT";
    case CUDACC_ERROR_INVALID_OPTION:
      return "CUDACC_ERROR_INVALID_OPTION";
    case CUDACC_ERROR_COMPILATION:
      return "CUDACC_ERROR_COMPILATION";
    case CUDACC_ERROR_LINKING:
      return "CUDACC_ERROR_LINKING";
    case CUDACC_ERROR_PCH_CREATE:
      return "CUDACC_ERROR_PCH_CREATE";
    case CUDACC_ERROR_INTERNAL_ERROR:
      return "CUDACC_ERROR_INTERNAL_ERROR";
  }
  return "CUDACC_ERROR_UNKNOWN";
}

extern "C" cudaccResult cudaccDestroyOutput(cudaccOutput* output)
{
  if (!output)
  {
    return CUDACC_SUCCESS;
  }
  freeOutput(*output);
  return CUDACC_SUCCESS;
}

extern "C" cudaccResult cudaccCompile(cudaccOutput* output, const char** options, size_t num_options)
{
  if (!output)
  {
    return CUDACC_ERROR_INVALID_INPUT;
  }

  initOutput(*output);

  cudacc::CompileRequest request;
  std::string diagnostics;
  if (!cudacc::parseOptions(num_options, options, request, diagnostics))
  {
    return finishCompile(*output, CUDACC_ERROR_INVALID_OPTION, "Option error: " + diagnostics);
  }

  cudaccResult validation_result = CUDACC_ERROR_INVALID_INPUT;
  if (!cudacc::validateCompileRequest(request, diagnostics, validation_result))
  {
    return finishCompile(*output, validation_result, diagnostics);
  }

  cudacc::initialize_llvm();
  cudacc::CompilerImpl compiler;

  switch (request.output_kind)
  {
    case cudacc::OutputKind::device_ir:
    {
      auto result = compiler.compileToDeviceBitcode(request.source, request.source_name, request.config);
      diagnostics += result.diagnostics;
      if (!result)
      {
        return finishCompile(*output, CUDACC_ERROR_COMPILATION, diagnostics);
      }
      setOutputData(*output, std::span<const char>{result.bitcode.data(), result.bitcode.size()});
      return finishCompile(*output, CUDACC_SUCCESS, diagnostics);
    }

    case cudacc::OutputKind::ptx:
    {
      auto result = compiler.compileToPTX(request.source, request.source_name, request.device_ir_inputs, request.config);
      diagnostics += result.diagnostics;
      if (!result)
      {
        return finishCompile(*output, CUDACC_ERROR_COMPILATION, diagnostics);
      }
      setOutputData(*output, std::span<const char>{result.ptx.data(), result.ptx.size()});
      return finishCompile(*output, CUDACC_SUCCESS, diagnostics);
    }

    case cudacc::OutputKind::cubin:
    {
      std::vector<char> generated_ptx;
      if (request.has_source)
      {
        auto ptx_result = compiler.compileToPTX(request.source, request.source_name, request.device_ir_inputs, request.config);
        diagnostics += ptx_result.diagnostics;
        if (!ptx_result)
        {
          return finishCompile(*output, CUDACC_ERROR_COMPILATION, diagnostics);
        }
        generated_ptx = std::move(ptx_result.ptx);
      }

      auto result = compiler.linkToCubin(
        std::span<const char>{generated_ptx.data(), generated_ptx.size()},
        request.ptx_inputs,
        request.cubin_inputs,
        request.ltoir_inputs,
        request.config);
      diagnostics += result.diagnostics;
      if (!result)
      {
        return finishCompile(*output, CUDACC_ERROR_COMPILATION, diagnostics);
      }
      setOutputData(*output, std::span<const char>{result.cubin.data(), result.cubin.size()});
      return finishCompile(*output, CUDACC_SUCCESS, diagnostics);
    }

    case cudacc::OutputKind::fatbin:
    {
      std::vector<char> generated_ptx;
      std::vector<char> generated_cubin;
      if (request.has_source)
      {
        auto ptx_result = compiler.compileToPTX(request.source, request.source_name, request.device_ir_inputs, request.config);
        diagnostics += ptx_result.diagnostics;
        if (!ptx_result)
        {
          return finishCompile(*output, CUDACC_ERROR_COMPILATION, diagnostics);
        }
        generated_ptx = std::move(ptx_result.ptx);

        auto cubin_result = compiler.linkToCubin(
          std::span<const char>{generated_ptx.data(), generated_ptx.size()}, {}, {}, request.ltoir_inputs, request.config);
        diagnostics += cubin_result.diagnostics;
        if (!cubin_result)
        {
          return finishCompile(*output, CUDACC_ERROR_COMPILATION, diagnostics);
        }
        generated_cubin = std::move(cubin_result.cubin);
      }

      auto result = compiler.createFatbin(
        std::span<const char>{generated_ptx.data(), generated_ptx.size()},
        std::span<const char>{generated_cubin.data(), generated_cubin.size()},
        request.ptx_inputs,
        request.cubin_inputs,
        request.ltoir_inputs,
        request.config);
      diagnostics += result.diagnostics;
      if (!result)
      {
        return finishCompile(*output, CUDACC_ERROR_INTERNAL_ERROR, diagnostics);
      }
      setOutputData(*output, std::span<const char>{result.fatbin.data(), result.fatbin.size()});
      return finishCompile(*output, CUDACC_SUCCESS, diagnostics);
    }

    case cudacc::OutputKind::create_device_pch:
    case cudacc::OutputKind::create_host_pch:
    {
      const bool is_device = request.output_kind == cudacc::OutputKind::create_device_pch;
      const bool success   = compiler.createPCH(
        request.source,
        is_device,
        request.pch_source_path,
        request.output_path,
        request.config,
        diagnostics);
      return finishCompile(*output, success ? CUDACC_SUCCESS : CUDACC_ERROR_PCH_CREATE, diagnostics);
    }

    case cudacc::OutputKind::compile:
    {
      auto result = compiler.compileToObject(
        request.source,
        request.source_name,
        request.output_path,
        request.device_ir_inputs,
        request.ltoir_inputs,
        request.config);
      diagnostics += result.diagnostics;
      if (!result)
      {
        return finishCompile(*output, CUDACC_ERROR_COMPILATION, diagnostics);
      }
      setOutputData(*output, std::span<const char>{result.cubin.data(), result.cubin.size()});
      return finishCompile(*output, CUDACC_SUCCESS, diagnostics);
    }

    case cudacc::OutputKind::shared:
    {
      auto result = compiler.linkToSharedLibrary(request.link_input_paths, request.output_path, request.config);
      diagnostics += result.diagnostics;
      return finishCompile(*output, result ? CUDACC_SUCCESS : CUDACC_ERROR_LINKING, diagnostics);
    }

    case cudacc::OutputKind::none:
      break;
  }

  return finishCompile(*output, CUDACC_ERROR_INTERNAL_ERROR, "Unhandled output kind");
}
