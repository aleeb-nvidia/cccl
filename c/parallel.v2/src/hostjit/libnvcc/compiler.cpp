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
#include <libnvcc/libnvcc.h>
#include <lld/Common/Driver.h>
#include <llvm/ADT/ScopeExit.h>
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
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

// Selective target initialization (native host target plus NVPTX for device)
extern "C" {
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

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <nvFatbin.h>
#include <nvJitLink.h>

namespace libnvcc
{
static bool llvm_initialized = false;

static void initialize_llvm()
{
  if (llvm_initialized)
  {
    return;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  llvm_initialized = true;
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
  std::vector<std::string> device_bitcode_files;
  std::vector<std::string> device_ltoir_files;
  std::unordered_map<std::string, std::string> macro_definitions;
  std::vector<std::string> extra_clang_args;
  int sm_version         = 75;
  int optimization_level = 2;
  bool debug             = false;
  bool verbose           = false;
  bool trace_includes    = false;
  bool keep_artifacts    = false;
};

struct CompilationResult
{
  bool success = false;
  std::string object_file_path;
  std::string diagnostics;

  explicit operator bool() const
  {
    return success;
  }
};

struct BitcodeResult
{
  bool success = false;
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
                      llvm::ArrayRef<char> data,
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

static bool parseOptions(int num_options, const char* const* raw_options, CompilerOptions& options, std::string& error)
{
  if (num_options < 0)
  {
    error = "Option count must be non-negative";
    return false;
  }
  if (num_options > 0 && raw_options == nullptr)
  {
    error = "Options array is null";
    return false;
  }

  setDefaultOptions(options);

  auto value_after_equals = [](std::string_view option, std::string_view prefix) -> std::string {
    return std::string(option.substr(prefix.size()));
  };

  for (int i = 0; i < num_options; ++i)
  {
    if (raw_options[i] == nullptr)
    {
      error = "Option string is null";
      return false;
    }

    std::string_view option(raw_options[i]);
    if (option.starts_with("--cuda-path="))
    {
      options.cuda_toolkit_path = value_after_equals(option, "--cuda-path=");
    }
    else if (option.starts_with("--hostjit-include-path="))
    {
      options.hostjit_include_path = value_after_equals(option, "--hostjit-include-path=");
    }
    else if (option.starts_with("--clang-headers-path="))
    {
      options.clang_headers_path = value_after_equals(option, "--clang-headers-path=");
    }
    else if (option.starts_with("--system-include-path="))
    {
      options.system_include_paths.push_back(value_after_equals(option, "--system-include-path="));
    }
    else if (option.starts_with("-isystem") && option.size() > 8)
    {
      options.system_include_paths.emplace_back(option.substr(8));
    }
    else if (option == "-isystem")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-isystem requires an argument";
        return false;
      }
      options.system_include_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--include-path="))
    {
      options.include_paths.push_back(value_after_equals(option, "--include-path="));
    }
    else if (option.starts_with("-I") && option.size() > 2)
    {
      options.include_paths.emplace_back(option.substr(2));
    }
    else if (option == "-I")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-I requires an argument";
        return false;
      }
      options.include_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--library-path="))
    {
      options.library_paths.push_back(value_after_equals(option, "--library-path="));
    }
    else if (option.starts_with("-L") && option.size() > 2)
    {
      options.library_paths.emplace_back(option.substr(2));
    }
    else if (option == "-L")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-L requires an argument";
        return false;
      }
      options.library_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--device-bitcode="))
    {
      options.device_bitcode_files.push_back(value_after_equals(option, "--device-bitcode="));
    }
    else if (option.starts_with("--device-ltoir="))
    {
      options.device_ltoir_files.push_back(value_after_equals(option, "--device-ltoir="));
    }
    else if (option.starts_with("--define-macro="))
    {
      if (!parseMacroDefinition(value_after_equals(option, "--define-macro="), options))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-D") && option.size() > 2)
    {
      if (!parseMacroDefinition(std::string(option.substr(2)), options))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option == "-D")
    {
      if (++i >= num_options || raw_options[i] == nullptr || !parseMacroDefinition(raw_options[i], options))
      {
        error = "-D requires a macro definition";
        return false;
      }
    }
    else if (option.starts_with("--gpu-architecture="))
    {
      if (!parseGpuArchitecture(value_after_equals(option, "--gpu-architecture="), options.sm_version))
      {
        error = "Invalid GPU architecture: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("--optimization-level="))
    {
      if (!parseInt(value_after_equals(option, "--optimization-level="), options.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-O") && option.size() > 2)
    {
      if (!parseInt(std::string(option.substr(2)), options.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option == "--debug")
    {
      options.debug = true;
    }
    else if (option == "--verbose")
    {
      options.verbose = true;
    }
    else if (option == "--trace-includes")
    {
      options.trace_includes = true;
    }
    else if (option == "--keep-artifacts")
    {
      options.keep_artifacts = true;
    }
    else if (option.starts_with("--entry-point="))
    {
      options.entry_point_name = value_after_equals(option, "--entry-point=");
    }
    else if (option.starts_with("--device-pch="))
    {
      options.device_pch_path = value_after_equals(option, "--device-pch=");
    }
    else if (option.starts_with("--host-pch="))
    {
      options.host_pch_path = value_after_equals(option, "--host-pch=");
    }
    else if (option.starts_with("-XClang="))
    {
      options.extra_clang_args.emplace_back(option.substr(8));
    }
    else if (option == "-XClang")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-XClang requires an argument";
        return false;
      }
      options.extra_clang_args.emplace_back(raw_options[i]);
    }
    else
    {
      error = "Unknown option: " + std::string(option);
      return false;
    }
  }

  if (options.library_paths.empty())
  {
    addDefaultCudaLibraryPath(options);
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

  for (const auto& bitcode_path : options.device_bitcode_files)
  {
    if (!pathExists(bitcode_path))
    {
      if (error_message)
      {
        *error_message = "Device bitcode path does not exist: " + bitcode_path;
      }
      return false;
    }
  }

  for (const auto& ltoir_path : options.device_ltoir_files)
  {
    if (!pathExists(ltoir_path))
    {
      if (error_message)
      {
        *error_message = "Device LTOIR path does not exist: " + ltoir_path;
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
    // We do not have access to the windows CRT, but we are only running single threaded anyway.
    // Otherwise we have undefined symbols like _tls_index and _Init_thread_epoch.
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
    driver_arg_strings.front(), getHostTargetTriple(), diag_engine, "libnvcc clang driver", vfs);
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

  // Write preamble to a persistent file and generate a PCH from it.
  bool generatePCH(const std::string& pch_source,
                   const std::string& pch_source_path,
                   const std::string& pch_output_path,
                   std::vector<std::string> arg_strings,
                   const CompilerOptions& config,
                   const std::string& log_label,
                   std::string& diagnostics)
  {
    // Write preamble to the persistent source path
    if (!writeFile(
          pch_source_path,
          llvm::ArrayRef<char>{pch_source.data(), pch_source.size()},
          diagnostics,
          "Failed to write PCH preamble to " + pch_source_path))
    {
      return false;
    }

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
    compiler.getFrontendOpts().OutputFile = pch_output_path;

    clang::GeneratePCHAction pch_action;
    bool success = compiler.ExecuteAction(pch_action);

    setup->appendDiagnosticsTo(diagnostics);

    return success;
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  createVFSWithSource(const std::string& source_code, const std::string& virtual_path)
  {
    auto mem_fs = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    mem_fs->addFile(virtual_path, 0, llvm::MemoryBuffer::getMemBuffer(source_code));

    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(mem_fs);
    return overlay;
  }

  bool compileDeviceToPTX(
    const std::string& source_code,
    const std::string& input_file,
    const std::string& output_ptx,
    const CompilerOptions& config,
    std::string& diagnostics)
  {
    std::string temp_dir    = std::filesystem::path(output_ptx).parent_path().string();
    std::string source_file = temp_dir + "/" + input_file;

    std::vector<std::string> bitcode_files_to_link = config.device_bitcode_files;

    std::vector<std::string> driver_arg_strings;
    appendCommonClangDriverArgs(driver_arg_strings, source_file, output_ptx, config, /*is_device=*/true);

    auto setup = createClangCompilerSetup(
      std::move(driver_arg_strings),
      createVFSWithSource(source_code, source_file),
      config.device_pch_path,
      config,
      "Device",
      diagnostics,
      "\nFailed to create device compiler invocation");
    if (!setup)
    {
      return false;
    }

    auto& compiler = setup->compiler;
    compiler.getFrontendOpts().OutputFile = output_ptx;

    if (config.trace_includes)
    {
      diagnostics += "\n=== Device Header Search Paths ===\n";
      const auto& hso = setup->invocation().getHeaderSearchOpts();
      for (const auto& entry : hso.UserEntries)
      {
        diagnostics += "  " + entry.Path + "\n";
      }
      diagnostics += "=== End Header Search Paths ===\n\n";
    }

    llvm::LLVMContext llvm_context;

    clang::EmitLLVMOnlyAction emit_llvm_action(&llvm_context);
    bool success = compiler.ExecuteAction(emit_llvm_action);

    if (config.trace_includes && compiler.hasSourceManager())
    {
      diagnostics += "\n=== Device Included Files ===\n";
      auto& sm = compiler.getSourceManager();
      for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it)
      {
        diagnostics += "  " + it->first.getName().str() + "\n";
      }
      diagnostics += "=== End Included Files ===\n\n";
    }

    if (success)
    {
      std::unique_ptr<llvm::Module> mod = emit_llvm_action.takeModule();
      if (mod)
      {
        for (const auto& bc_file : bitcode_files_to_link)
        {
          llvm::SMDiagnostic err;
          auto bc_mod = llvm::parseIRFile(bc_file, err, llvm_context);
          if (bc_mod)
          {
            if (llvm::Linker::linkModules(*mod, std::move(bc_mod)))
            {
              diagnostics += "Failed to link bitcode: " + bc_file + "\n";
              success = false;
              break;
            }
          }
          else
          {
            std::string err_msg;
            llvm::raw_string_ostream err_stream(err_msg);
            err.print("hostjit", err_stream);
            diagnostics += "Failed to parse bitcode: " + bc_file + "\n" + err_msg + "\n";
            success = false;
            break;
          }
        }

        // Re-link libdevice to resolve any new references (e.g. __nv_pow)
        // introduced by the extra bitcode modules.
        if (success && !bitcode_files_to_link.empty())
        {
          std::string libdevice_path = config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc";
          llvm::SMDiagnostic err;
          auto libdevice = llvm::parseIRFile(libdevice_path, err, llvm_context);
          if (libdevice)
          {
            // Use AppendToUsed to avoid internalization issues
            llvm::Linker::linkModules(*mod, std::move(libdevice), llvm::Linker::LinkOnlyNeeded);
          }
        }

        if (success)
        {
          std::string err_str;
          const llvm::Target* target = llvm::TargetRegistry::lookupTarget(mod->getTargetTriple(), err_str);
          if (target)
          {
            llvm::TargetOptions opt;
            auto tm = target->createTargetMachine(
              mod->getTargetTriple(),
              "sm_" + std::to_string(config.sm_version),
              "+ptx" + std::to_string(ptxVersionForSM(config.sm_version)),
              opt,
              llvm::Reloc::PIC_);
            if (tm)
            {
              mod->setDataLayout(tm->createDataLayout());

              // Run optimization passes after linking to inline user-provided
              // operations (from bitcode or embedded C++ source).
              if (!config.entry_point_name.empty())
              {
                // Internalize all functions except the entry point and
                // GPU kernels, so the optimizer can inline the linked
                // bitcode functions.
                for (auto& F : *mod)
                {
                  if (!F.isDeclaration() && F.getLinkage() == llvm::GlobalValue::ExternalLinkage
                      && F.getName() != config.entry_point_name && F.getCallingConv() != llvm::CallingConv::PTX_Kernel)
                  {
                    F.setLinkage(llvm::GlobalValue::InternalLinkage);
                    // Remove attributes that conflict with inlining
                    F.removeFnAttr(llvm::Attribute::NoInline);
                    F.removeFnAttr(llvm::Attribute::OptimizeNone);
                    F.addFnAttr(llvm::Attribute::AlwaysInline);
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

                // Raise LLVM's loop-unroll thresholds (once) so the user op's
                // small, constant-trip-count loops -- e.g. Numba `local.array`
                // loops -- get FULLY unrolled. Without full unroll the backing
                // alloca keeps a dynamic index, SROA can't promote it, and it
                // lands in local memory (a per-thread stack frame + LDL/STL
                // traffic). ptxas does this promotion on the v1/LTO path; the
                // LLVM-NVPTX path needs full-unroll-then-SROA at the IR level.
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

                llvm::LoopAnalysisManager LAM;
                llvm::FunctionAnalysisManager FAM;
                llvm::CGSCCAnalysisManager CGAM;
                llvm::ModuleAnalysisManager MAM;

                llvm::PassBuilder PB(tm);
                PB.registerModuleAnalyses(MAM);
                PB.registerCGSCCAnalyses(CGAM);
                PB.registerFunctionAnalyses(FAM);
                PB.registerLoopAnalyses(LAM);
                PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

                auto MPM = PB.buildPerModuleDefaultPipeline(opt_level);
                MPM.run(*mod, MAM);

                // Second optimization round with fresh analyses: now that the
                // op's loops are fully unrolled (constant indices), the early
                // SROA in the pipeline promotes the local arrays to registers.
                llvm::LoopAnalysisManager LAM2;
                llvm::FunctionAnalysisManager FAM2;
                llvm::CGSCCAnalysisManager CGAM2;
                llvm::ModuleAnalysisManager MAM2;

                llvm::PassBuilder PB2(tm);
                PB2.registerModuleAnalyses(MAM2);
                PB2.registerCGSCCAnalyses(CGAM2);
                PB2.registerFunctionAnalyses(FAM2);
                PB2.registerLoopAnalyses(LAM2);
                PB2.crossRegisterProxies(LAM2, FAM2, CGAM2, MAM2);

                auto MPM2 = PB2.buildPerModuleDefaultPipeline(opt_level);
                MPM2.run(*mod, MAM2);
              }

              std::error_code EC;
              llvm::raw_fd_ostream dest(output_ptx, EC);
              if (!EC)
              {
                llvm::legacy::PassManager pass;
                tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::AssemblyFile);
                pass.run(*mod);
                dest.flush();

                // Debug: when LIBNVCC_DUMP_DIR is set, dump the optimized IR
                // and the PTX fed to ptxas, keyed by entry point name. Lets us
                // inspect codegen (register pressure, launch bounds) post-inline.
                if (const char* dump_dir = std::getenv("LIBNVCC_DUMP_DIR"))
                {
                  std::error_code dec;
                  std::filesystem::create_directories(dump_dir, dec);
                  const std::string base =
                    config.entry_point_name.empty() ? std::string("kernel") : config.entry_point_name;
                  const std::string stem = (std::filesystem::path(dump_dir) / base).string();
                  llvm::raw_fd_ostream ll_os(stem + ".opt.ll", dec);
                  if (!dec)
                  {
                    mod->print(ll_os, nullptr);
                  }
                  std::error_code cec;
                  std::filesystem::copy_file(
                    output_ptx, stem + ".ptx", std::filesystem::copy_options::overwrite_existing, cec);
                  llvm::errs() << "[hostjit] dumped " << stem << ".opt.ll and " << stem << ".ptx\n";
                }
              }
              else
              {
                diagnostics += "Failed to open output file: " + output_ptx + "\n";
                success = false;
              }
            }
            else
            {
              diagnostics += "Failed to create target machine\n";
              success = false;
            }
          }
          else
          {
            diagnostics += "Failed to lookup target: " + err_str + "\n";
            success = false;
          }
        }
      }
    }

    setup->appendDiagnosticsTo(diagnostics);

    return success;
  }

  BitcodeResult compileToDeviceBitcode(
    const std::string& source_code,
    const std::string& input_name,
    const std::string& output_bitcode_path,
    const CompilerOptions& config)
  {
    BitcodeResult result;

    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      result.diagnostics = "Configuration error: " + error_msg;
      return result;
    }

    initialize_llvm();

    std::string temp_dir =
      (tempDirectoryPath() / ("hostjit_bc_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }
    auto cleanup_temp_dir = llvm::scope_exit([&] {
      removeAll(temp_dir);
    });

    std::string input_file   = input_name.empty() ? std::string("input.cu") : input_name;
    std::string source_file  = temp_dir + "/" + input_file;

    std::vector<std::string> driver_arg_strings;
    appendCommonClangDriverArgs(
      driver_arg_strings, source_file, output_bitcode_path, config, /*is_device=*/true);

    auto setup = createClangCompilerSetup(
      std::move(driver_arg_strings),
      createVFSWithSource(source_code, source_file),
      config.device_pch_path,
      config,
      "Device bitcode",
      result.diagnostics,
      "\nFailed to create compiler invocation");
    if (!setup)
    {
      return result;
    }

    auto& compiler = setup->compiler;

    llvm::LLVMContext llvm_context;
    clang::EmitLLVMOnlyAction emit_llvm_action(&llvm_context);
    bool success = compiler.ExecuteAction(emit_llvm_action);

    if (success)
    {
      std::unique_ptr<llvm::Module> mod = emit_llvm_action.takeModule();
      if (mod)
      {
        std::error_code ec;
        llvm::raw_fd_ostream os(output_bitcode_path, ec, llvm::sys::fs::OF_None);
        if (ec)
        {
          result.diagnostics = "Failed to open bitcode output file: " + output_bitcode_path + "\n";
        }
        else
        {
          llvm::WriteBitcodeToFile(*mod, os);
          os.flush();
          if (os.has_error())
          {
            result.diagnostics = "Failed to write bitcode output file: " + output_bitcode_path + "\n";
          }
          else
          {
            result.success = true;
          }
        }
      }
      else
      {
        result.diagnostics = "Failed to get LLVM module";
      }
    }

    setup->appendDiagnosticsTo(result.diagnostics);
    if (config.keep_artifacts)
    {
      cleanup_temp_dir.release();
    }
    return result;
  }

  bool compileHostCode(
    const std::string& source_code,
    const std::string& input_file,
    const std::string& fatbin_path,
    const std::string& output_obj,
    const CompilerOptions& config,
    std::string& diagnostics)
  {
    std::string temp_dir    = std::filesystem::path(output_obj).parent_path().string();
    std::string source_file = temp_dir + "/host_" + input_file;

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
    bool success = compiler.ExecuteAction(emit_action);

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

  CompilationResult compileToObject(
    const std::string& source_code,
    const std::string& input_name,
    const std::string& output_path,
    const std::string& output_cubin_path,
    const CompilerOptions& config)
  {
    CompilationResult result;
    result.success          = false;
    result.object_file_path = output_path;

    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      result.diagnostics = "Configuration error: " + error_msg;
      return result;
    }

    initialize_llvm();

    std::string temp_dir =
      (tempDirectoryPath() / ("hostjit_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }
    auto cleanup_temp_dir = llvm::scope_exit([&] {
      removeAll(temp_dir);
    });

    std::string input_file  = input_name.empty() ? std::string("input.cu") : input_name;
    std::string ptx_file    = temp_dir + "/device.ptx";
    std::string fatbin_file = temp_dir + "/device.fatbin";

    if (config.verbose)
    {
      result.diagnostics += "=== Device compilation ===\n";
    }

    if (!compileDeviceToPTX(source_code, input_file, ptx_file, config, result.diagnostics))
    {
      result.diagnostics += "\nDevice compilation failed";
      return result;
    }

    if (config.verbose)
    {
      result.diagnostics += "\n=== nvJitLink + fatbinary ===\n";
    }

    {
      std::vector<char> ptx_data;
      {
        std::ifstream f(ptx_file, std::ios::binary);
        ptx_data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
      }
      if (ptx_data.empty())
      {
        result.diagnostics += "\nFailed to read ptx file";
        return result;
      }
      if (ptx_data.back() != '\0')
      {
        ptx_data.push_back('\0');
      }

      std::string arch_opt  = "-arch=sm_" + std::to_string(config.sm_version);
      std::string opt_level = "-O" + std::to_string(config.optimization_level >= 1 ? 3 : 0);
      std::vector<std::string> jitlink_option_strs{arch_opt, opt_level};
      // LTOIR inputs require -lto. When present, both the PTX and the LTOIRs
      // get linked through the LTO codegen path.
      const bool have_ltoir = !config.device_ltoir_files.empty();
      if (have_ltoir)
      {
        jitlink_option_strs.emplace_back("-lto");
      }
      std::vector<const char*> jitlink_options;
      jitlink_options.reserve(jitlink_option_strs.size());
      for (const auto& s : jitlink_option_strs)
      {
        jitlink_options.push_back(s.c_str());
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

      jlr = nvJitLinkAddData(jitlink_handle, NVJITLINK_INPUT_PTX, ptx_data.data(), ptx_data.size(), "device.ptx");
      if (jlr != NVJITLINK_SUCCESS)
      {
        size_t log_size = 0;
        nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
        if (log_size > 1)
        {
          std::string log(log_size, '\0');
          nvJitLinkGetErrorLog(jitlink_handle, log.data());
          result.diagnostics += "\n" + log;
        }
        result.diagnostics += "\nnvJitLinkAddData failed";
        return result;
      }

      // Feed LTO-IR inputs to nvJitLink alongside the device PTX. This is the
      // escape-hatch path for callers with pre-built nvcc -dlto artifacts;
      // Python-emitted user ops travel as LLVM bitcode through the path above
      // and are already inlined into the PTX by the time we get here.
      // nvJitLink resolves any remaining extern symbol(s) from these modules.
      for (const auto& ltoir_path : config.device_ltoir_files)
      {
        std::ifstream f(ltoir_path, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (buf.empty())
        {
          continue;
        }
        jlr = nvJitLinkAddData(jitlink_handle, NVJITLINK_INPUT_LTOIR, buf.data(), buf.size(), ltoir_path.c_str());
        if (jlr != NVJITLINK_SUCCESS)
        {
          size_t log_size = 0;
          nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
          if (log_size > 1)
          {
            std::string log(log_size, '\0');
            nvJitLinkGetErrorLog(jitlink_handle, log.data());
            result.diagnostics += "\n" + log;
          }
          result.diagnostics += "\nnvJitLinkAddData(LTOIR) failed for " + ltoir_path;
          return result;
        }
      }

      jlr = nvJitLinkComplete(jitlink_handle);
      if (jlr != NVJITLINK_SUCCESS)
      {
        size_t log_size = 0;
        nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
        if (log_size > 1)
        {
          std::string log(log_size, '\0');
          nvJitLinkGetErrorLog(jitlink_handle, log.data());
          result.diagnostics += "\n" + log;
        }
        result.diagnostics += "\nnvJitLinkComplete failed";
        return result;
      }

      size_t cubin_size = 0;
      nvJitLinkGetLinkedCubinSize(jitlink_handle, &cubin_size);
      std::vector<char> cubin_data(cubin_size);
      nvJitLinkGetLinkedCubin(jitlink_handle, cubin_data.data());

      if (!output_cubin_path.empty())
      {
        if (!writeFile(
              output_cubin_path,
              llvm::ArrayRef<char>{cubin_data.data(), cubin_data.size()},
              result.diagnostics,
              "\nFailed to write cubin file"))
        {
          return result;
        }
      }

      std::string arch             = std::to_string(config.sm_version);
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

      fbr = nvFatbinAddCubin(fatbin_handle, cubin_data.data(), cubin_data.size(), arch.c_str(), "device.cubin");
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddCubin failed: ") + nvFatbinGetErrorString(fbr);
        return result;
      }

      fbr = nvFatbinAddPTX(fatbin_handle, ptx_data.data(), ptx_data.size(), arch.c_str(), "device.ptx", nullptr);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddPTX failed: ") + nvFatbinGetErrorString(fbr);
        return result;
      }

      size_t fatbin_size = 0;
      fbr                = nvFatbinSize(fatbin_handle, &fatbin_size);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinSize failed: ") + nvFatbinGetErrorString(fbr);
        return result;
      }

      std::vector<char> fatbin_data(fatbin_size);
      fbr = nvFatbinGet(fatbin_handle, fatbin_data.data());
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinGet failed: ") + nvFatbinGetErrorString(fbr);
        return result;
      }

      if (!writeFile(
            fatbin_file,
            llvm::ArrayRef<char>{fatbin_data.data(), fatbin_data.size()},
            result.diagnostics,
            "\nFailed to write fatbin file"))
      {
        return result;
      }
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
    result.success = true;
    return result;
  }

  bool createPCH(const std::string& source_code,
                 libnvccPCHKind kind,
                 const std::string& pch_source_path,
                 const std::string& pch_output_path,
                 const CompilerOptions& config,
                 std::string& diagnostics)
  {
    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      diagnostics = "Configuration error: " + error_msg;
      return false;
    }

    if (kind != LIBNVCC_PCH_DEVICE && kind != LIBNVCC_PCH_HOST)
    {
      diagnostics = "Invalid PCH kind";
      return false;
    }

    initialize_llvm();

    std::vector<std::string> arg_strings;
    appendCommonClangDriverArgs(
      arg_strings, pch_source_path, pch_output_path, config, /*is_device=*/kind == LIBNVCC_PCH_DEVICE);
    return generatePCH(
      source_code,
      pch_source_path,
      pch_output_path,
      std::move(arg_strings),
      config,
      kind == LIBNVCC_PCH_DEVICE ? "Device PCH" : "Host PCH",
      diagnostics);
  }

  LinkResult linkToSharedLibrary(
    const std::vector<std::string>& object_files, const std::string& output_path, const CompilerOptions& config)
  {
    LinkResult result;
    result.success      = false;
    result.library_path = output_path;

    if (object_files.empty())
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

    // Generate import libraries from DLLs present on the system,
    // so we don't require the Windows SDK or MSVC .lib files.
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

    for (const auto& obj_file : object_files)
    {
      arg_strings.push_back(obj_file);
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
    // Allow unresolved symbols — they will be satisfied at dlopen() time
    // by libraries already loaded in the host process (libc, libstdc++,
    // cudart, etc.).  This removes the need for system CRT objects and
    // dev packages on the target machine.
    arg_strings.push_back("--allow-shlib-undefined");
    arg_strings.push_back("-o");
    arg_strings.push_back(output_path);

    for (const auto& lib_path : config.library_paths)
    {
      arg_strings.push_back("-L" + lib_path);
      // Embed the library path as RPATH so the dynamic linker can find
      // libcudart.so.XX at dlopen time without LD_LIBRARY_PATH.
      arg_strings.push_back("-rpath");
      arg_strings.push_back(lib_path);
    }

    for (const auto& obj_file : object_files)
    {
      arg_strings.push_back(obj_file);
    }

    // pip packages ship libcudart.so.XX without an unversioned symlink,
    // so -lcudart won't work.  Find the actual .so by scanning library_paths.
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
} // namespace libnvcc

struct libnvccProgram_st
{
  std::string source;
  std::string name;
  std::string log;
  libnvcc::CompilerImpl compiler;
};

namespace
{
void setProgramLog(libnvccProgram prog, std::string log)
{
  if (prog)
  {
    prog->log = std::move(log);
  }
}

bool parseProgramOptions(
  libnvccProgram prog, int num_options, const char* const* raw_options, libnvcc::CompilerOptions& options)
{
  std::string error;
  if (!libnvcc::parseOptions(num_options, raw_options, options, error))
  {
    setProgramLog(prog, "Option error: " + error);
    return false;
  }
  return true;
}
} // anonymous namespace

extern "C" const char* libnvccGetErrorString(libnvccResult result)
{
  switch (result)
  {
    case LIBNVCC_SUCCESS:
      return "LIBNVCC_SUCCESS";
    case LIBNVCC_ERROR_OUT_OF_MEMORY:
      return "LIBNVCC_ERROR_OUT_OF_MEMORY";
    case LIBNVCC_ERROR_PROGRAM_CREATION_FAILURE:
      return "LIBNVCC_ERROR_PROGRAM_CREATION_FAILURE";
    case LIBNVCC_ERROR_INVALID_INPUT:
      return "LIBNVCC_ERROR_INVALID_INPUT";
    case LIBNVCC_ERROR_INVALID_PROGRAM:
      return "LIBNVCC_ERROR_INVALID_PROGRAM";
    case LIBNVCC_ERROR_INVALID_OPTION:
      return "LIBNVCC_ERROR_INVALID_OPTION";
    case LIBNVCC_ERROR_COMPILATION:
      return "LIBNVCC_ERROR_COMPILATION";
    case LIBNVCC_ERROR_LINKING:
      return "LIBNVCC_ERROR_LINKING";
    case LIBNVCC_ERROR_PCH_CREATE:
      return "LIBNVCC_ERROR_PCH_CREATE";
    case LIBNVCC_ERROR_INTERNAL_ERROR:
      return "LIBNVCC_ERROR_INTERNAL_ERROR";
  }
  return "LIBNVCC_ERROR_UNKNOWN";
}

extern "C" libnvccResult libnvccCreateProgram(libnvccProgram* prog, const char* src, const char* name)
{
  if (!prog || !src)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  *prog = nullptr;

  auto* program   = new libnvccProgram_st;
  program->source = src;
  program->name   = (name && name[0]) ? name : "input.cu";
  *prog           = program;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccDestroyProgram(libnvccProgram* prog)
{
  if (!prog || !*prog)
  {
    return LIBNVCC_SUCCESS;
  }
  delete *prog;
  *prog = nullptr;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccCompileProgramToDeviceBitcode(
  libnvccProgram prog, const char* outputBitcodePath, int numOptions, const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!outputBitcodePath || outputBitcodePath[0] == '\0')
  {
    setProgramLog(prog, "outputBitcodePath must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  auto result = prog->compiler.compileToDeviceBitcode(prog->source, prog->name, outputBitcodePath, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_COMPILATION;
}

extern "C" libnvccResult libnvccCompileProgramToObject(
  libnvccProgram prog,
  const char* outputObjectPath,
  const char* outputCubinPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!outputObjectPath || outputObjectPath[0] == '\0')
  {
    setProgramLog(prog, "outputObjectPath must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  const std::string cubin_path = outputCubinPath ? outputCubinPath : "";
  auto result = prog->compiler.compileToObject(prog->source, prog->name, outputObjectPath, cubin_path, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_COMPILATION;
}

extern "C" libnvccResult libnvccLinkToSharedLibrary(
  libnvccProgram prog,
  int numObjectFiles,
  const char* const* objectFiles,
  const char* outputLibraryPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (numObjectFiles < 0 || (numObjectFiles > 0 && !objectFiles) || !outputLibraryPath || outputLibraryPath[0] == '\0')
  {
    setProgramLog(prog, "Invalid link input");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  std::vector<std::string> object_files;
  object_files.reserve(static_cast<size_t>(numObjectFiles));
  for (int i = 0; i < numObjectFiles; ++i)
  {
    if (!objectFiles[i] || objectFiles[i][0] == '\0')
    {
      setProgramLog(prog, "Object file path must be non-empty");
      return LIBNVCC_ERROR_INVALID_INPUT;
    }
    object_files.emplace_back(objectFiles[i]);
  }

  auto result = prog->compiler.linkToSharedLibrary(object_files, outputLibraryPath, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_LINKING;
}

extern "C" libnvccResult libnvccCreatePCH(
  libnvccProgram prog,
  libnvccPCHKind kind,
  const char* pchSourcePath,
  const char* pchOutputPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!pchSourcePath || pchSourcePath[0] == '\0' || !pchOutputPath || pchOutputPath[0] == '\0')
  {
    setProgramLog(prog, "PCH source and output paths must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  std::string diagnostics;
  bool success =
    prog->compiler.createPCH(prog->source, kind, pchSourcePath, pchOutputPath, parsed_options, diagnostics);
  setProgramLog(prog, diagnostics);
  return success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_PCH_CREATE;
}

extern "C" libnvccResult libnvccGetProgramLogSize(libnvccProgram prog, size_t* logSizeRet)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!logSizeRet)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  *logSizeRet = prog->log.size() + 1;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccGetProgramLog(libnvccProgram prog, char* log)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!log)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  std::memcpy(log, prog->log.c_str(), prog->log.size() + 1);
  return LIBNVCC_SUCCESS;
}
