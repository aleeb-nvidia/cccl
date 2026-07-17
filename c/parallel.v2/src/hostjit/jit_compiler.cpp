#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <hostjit/jit_compiler.hpp>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <objbase.h>
#  include <windows.h>
#else
#  include <stdlib.h>
#endif

namespace
{
static constexpr const char* pch_preamble_source =
  "#include <cuda_runtime.h>\n"
  "#include <cuda/std/iterator>\n"
  "#include <cuda/std/functional>\n"
  "#include <cuda/functional>\n"
  "#include <cub/device/device_adjacent_difference.cuh>\n"
  "#include <cub/device/device_copy.cuh>\n"
  "#include <cub/device/device_find.cuh>\n"
  "#include <cub/device/device_for.cuh>\n"
  "#include <cub/device/device_histogram.cuh>\n"
  "#include <cub/device/device_merge.cuh>\n"
  "#include <cub/device/device_merge_sort.cuh>\n"
  "#include <cub/device/device_partition.cuh>\n"
  "#include <cub/device/device_radix_sort.cuh>\n"
  "#include <cub/device/device_reduce.cuh>\n"
  "#include <cub/device/device_scan.cuh>\n"
  "#include <cub/device/device_segmented_radix_sort.cuh>\n"
  "#include <cub/device/device_segmented_scan.cuh>\n"
  "#include <cub/device/device_segmented_sort.cuh>\n"
  "#include <cub/device/device_select.cuh>\n"
  "#include <cub/device/device_transform.cuh>\n";

std::filesystem::path get_pch_cache_dir()
{
  auto dir = std::filesystem::temp_directory_path() / "hostjit_pch";
  std::filesystem::create_directories(dir);
  return dir;
}

std::string get_pch_path(const std::string& kind, int sm_version)
{
  return (get_pch_cache_dir() / (kind + "_sm" + std::to_string(sm_version) + ".pch")).string();
}

std::string get_pch_source_path(const std::string& kind, int sm_version)
{
  return (get_pch_cache_dir() / (kind + "_sm" + std::to_string(sm_version) + "_preamble.cu")).string();
}

bool create_pch_if_needed(
  hostjit::CompilerConfig config,
  cudaccPCHKind kind,
  const std::string& kind_name,
  std::string& diagnostics,
  std::string& pch_path)
{
  pch_path = get_pch_path(kind_name, config.sm_version);
  if (std::filesystem::exists(pch_path))
  {
    return true;
  }

  config.enable_pch = false;
  config.device_pch_path.clear();
  config.host_pch_path.clear();

  std::vector<std::string> options;
  config.appendCommandLineArguments(options);
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  hostjit::detail::CudaccProgramGuard program;
  auto create_result = cudaccCreateProgram(&program.program, pch_preamble_source, "hostjit_preamble.cu");
  if (create_result != CUDACC_SUCCESS)
  {
    diagnostics += "Failed to create cudacc PCH program: ";
    diagnostics += cudaccGetErrorString(create_result);
    diagnostics += "\n";
    pch_path.clear();
    return false;
  }

  auto source_path = get_pch_source_path(kind_name, config.sm_version);
  auto pch_result  = cudaccCreatePCH(
    program.program,
    kind,
    source_path.c_str(),
    pch_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  if (pch_result != CUDACC_SUCCESS)
  {
    diagnostics += kind_name + " PCH generation failed: " + hostjit::detail::get_cudacc_program_log(program.program);
    diagnostics += "\n";
    pch_path.clear();
    return false;
  }
  return true;
}

hostjit::CompilerConfig prepare_pch_config(const hostjit::CompilerConfig& config, std::string& diagnostics)
{
  hostjit::CompilerConfig prepared = config;
  prepared.device_pch_path.clear();
  prepared.host_pch_path.clear();

  if (!prepared.enable_pch)
  {
    return prepared;
  }

  std::string device_pch_path;
  if (create_pch_if_needed(prepared, CUDACC_PCH_DEVICE, "device", diagnostics, device_pch_path))
  {
    prepared.device_pch_path = std::move(device_pch_path);
  }

  std::string host_pch_path;
  if (create_pch_if_needed(prepared, CUDACC_PCH_HOST, "host", diagnostics, host_pch_path))
  {
    prepared.host_pch_path = std::move(host_pch_path);
  }

  return prepared;
}

bool read_file(const std::filesystem::path& path, std::vector<char>& out)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
  {
    return false;
  }
  auto size = f.tellg();
  if (size < 0)
  {
    return false;
  }

  out.resize(static_cast<size_t>(size));
  f.seekg(0);
  if (!out.empty())
  {
    f.read(out.data(), static_cast<std::streamsize>(out.size()));
  }
  return static_cast<bool>(f);
}

#ifdef _WIN32
std::string guid_to_string(const GUID& guid)
{
  char buffer[33]{};
  std::snprintf(buffer,
                sizeof(buffer),
                "%08lx%04x%04x%02x%02x%02x%02x%02x%02x%02x%02x",
                static_cast<unsigned long>(guid.Data1),
                guid.Data2,
                guid.Data3,
                guid.Data4[0],
                guid.Data4[1],
                guid.Data4[2],
                guid.Data4[3],
                guid.Data4[4],
                guid.Data4[5],
                guid.Data4[6],
                guid.Data4[7]);
  return buffer;
}
#endif
} // anonymous namespace

namespace hostjit
{
JITCompiler::JITCompiler()
    : config_(detectDefaultConfig())
{}

JITCompiler::JITCompiler(const CompilerConfig& config)
    : config_(config)
{}

JITCompiler::~JITCompiler()
{
  cleanup();
}

bool JITCompiler::compile(const std::string& source_code)
{
  std::string config_error;
  if (!validateConfig(config_, &config_error))
  {
    last_error_      = "Configuration error: " + config_error;
    last_error_code_ = std::make_error_code(std::errc::invalid_argument);
    return false;
  }

  cleanup();

  temp_dir_ = createTempDirectory();
  if (temp_dir_.empty())
  {
    last_error_      = "Failed to create temporary directory";
    last_error_code_ = std::make_error_code(std::errc::io_error);
    return false;
  }

  std::string pch_diagnostics;
  CompilerConfig cudacc_config = prepare_pch_config(config_, pch_diagnostics);
  if (config_.verbose && !pch_diagnostics.empty())
  {
    std::cout << pch_diagnostics;
  }

  std::vector<std::string> options;
  cudacc_config.appendCommandLineArguments(options);
  auto option_ptrs = hostjit::detail::make_cudacc_option_ptrs(options);

  hostjit::detail::CudaccProgramGuard program;
  auto create_result = cudaccCreateProgram(&program.program, source_code.c_str(), "input.cu");
  if (create_result != CUDACC_SUCCESS)
  {
    last_error_      = std::string("Failed to create cudacc program: ") + cudaccGetErrorString(create_result);
    last_error_code_ = std::make_error_code(std::errc::io_error);
    removeTempDirectory();
    return false;
  }

  std::filesystem::path obj_path   = temp_dir_ / "cuda_code.o";
  std::filesystem::path cubin_path = temp_dir_ / "device.cubin";
  std::string obj_path_string      = obj_path.string();
  std::string cubin_path_string    = cubin_path.string();
  auto compile_result    = cudaccCompileProgramToObject(
    program.program,
    obj_path_string.c_str(),
    cubin_path_string.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  auto compile_log = hostjit::detail::get_cudacc_program_log(program.program);

  if (compile_result != CUDACC_SUCCESS)
  {
    last_error_      = "Compilation failed:\n" + compile_log;
    last_error_code_ = std::make_error_code(std::errc::io_error);
    removeTempDirectory();
    return false;
  }

  cubin_.clear();
  if (!read_file(cubin_path, cubin_))
  {
    last_error_      = "Compilation failed: generated cubin could not be read";
    last_error_code_ = std::make_error_code(std::errc::io_error);
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Compilation diagnostics:\n" << compile_log << "\n";
  }

#ifdef _WIN32
  std::filesystem::path lib_path = temp_dir_ / "cuda_code.dll";
#else
  std::filesystem::path lib_path = temp_dir_ / "libcuda_code.so";
#endif
  std::string lib_path_string = lib_path.string();
  const char* object_files[]  = {obj_path_string.c_str()};
  auto link_result           = cudaccLinkToSharedLibrary(
    program.program,
    1,
    object_files,
    lib_path_string.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  auto link_log = hostjit::detail::get_cudacc_program_log(program.program);

  if (link_result != CUDACC_SUCCESS)
  {
    last_error_      = "Linking failed:\n" + link_log;
    last_error_code_ = std::make_error_code(std::errc::io_error);
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Linking diagnostics:\n" << link_log << "\n";
  }

  if (!library_.load(lib_path_string))
  {
    last_error_      = "Failed to load library: " + library_.getLastError();
    last_error_code_ = library_.getLastErrorCode();
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Successfully loaded library: " << lib_path_string << "\n";
  }

  last_error_.clear();
  last_error_code_.clear();
  return true;
}

void JITCompiler::cleanup()
{
  library_.unload();

  if (!config_.keep_artifacts)
  {
    removeTempDirectory();
  }

  last_error_.clear();
  last_error_code_.clear();
}

std::filesystem::path JITCompiler::createTempDirectory()
{
#ifdef _WIN32
  char temp_path[MAX_PATH + 1]{};
  DWORD len = GetTempPathA(static_cast<DWORD>(sizeof(temp_path)), temp_path);
  if (len == 0 || len > MAX_PATH)
  {
    return {};
  }

  GUID guid;
  if (CoCreateGuid(&guid) != S_OK)
  {
    return {};
  }
  std::filesystem::path full_path = std::filesystem::path(temp_path) / ("hostjit_" + guid_to_string(guid));
  if (!CreateDirectoryA(full_path.string().c_str(), nullptr))
  {
    return {};
  }
  return full_path;
#else
  std::error_code ec;
  std::filesystem::path base_tmp_dir = std::filesystem::temp_directory_path(ec);
  if (ec)
  {
    base_tmp_dir = "/tmp";
  }

  std::string path_template = (base_tmp_dir / "hostjit_XXXXXX").string();
  char* created = mkdtemp(path_template.data());
  if (!created)
  {
    return {};
  }

  return std::filesystem::path(created);
#endif
}

void JITCompiler::removeTempDirectory()
{
  if (temp_dir_.empty())
  {
    return;
  }

  std::error_code ec;
  std::filesystem::remove_all(temp_dir_, ec);
  if (ec && config_.verbose)
  {
    std::cerr << "Warning: Failed to remove temporary directory: " << ec.message() << "\n";
  }

  temp_dir_.clear();
}
} // namespace hostjit
