#pragma once

#include <cassert>
#include <string>
#include <vector>

#include <cudacc/cudacc.h>

namespace hostjit::detail
{
struct CudaccProgramGuard
{
  cudaccProgram program = nullptr;

  CudaccProgramGuard()                                      = default;
  CudaccProgramGuard(const CudaccProgramGuard&)            = delete;
  CudaccProgramGuard& operator=(const CudaccProgramGuard&) = delete;

  ~CudaccProgramGuard()
  {
    cudaccDestroyProgram(&program);
  }
};

inline std::vector<const char*> make_cudacc_option_ptrs(const std::vector<std::string>& options)
{
  std::vector<const char*> ptrs;
  ptrs.reserve(options.size());
  for (const auto& option : options)
  {
    ptrs.push_back(option.c_str());
  }
  return ptrs;
}

inline std::string get_cudacc_program_log(cudaccProgram program)
{
  size_t log_size = 0;
  if (cudaccGetProgramLogSize(program, &log_size) != CUDACC_SUCCESS)
  {
    return {};
  }

  assert(log_size > 0 && "Log size should include NUL terminator");
  if (log_size == 1)
  {
    return {};
  }

  std::string log(log_size, '\0');
  auto res = cudaccGetProgramLog(program, log.data());
  assert(res == CUDACC_SUCCESS && "Copying the log failed even though size calculation succeeded?");
  assert(log.back() == '\0' && "cudaccGetProgramLog() should append a NUL character");
  log.pop_back(); // Drop the extra NUL.
  return log;
}
} // namespace hostjit::detail
