#pragma once

#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cudacc/cudacc.h>

namespace hostjit::detail
{
struct CudaccOutputGuard
{
  cudaccOutput output{};

  CudaccOutputGuard()                                     = default;
  CudaccOutputGuard(const CudaccOutputGuard&)            = delete;
  CudaccOutputGuard& operator=(const CudaccOutputGuard&) = delete;

  ~CudaccOutputGuard()
  {
    cudaccDestroyOutput(&output);
  }
};

class CudaccOptionsBuilder
{
public:
  void add(std::string arg)
  {
    args_.push_back(Arg{std::move(arg), nullptr, false});
  }

  void add_pair(std::string flag, std::string value)
  {
    add(std::move(flag));
    add(std::move(value));
  }

  void add_memory_input(std::string flag, std::span<const char> data)
  {
    add(std::move(flag));
    add(std::to_string(data.size()));
    args_.push_back(Arg{{}, data.data(), true});
  }

  void add_all(const std::vector<std::string>& args)
  {
    for (const auto& arg : args)
    {
      add(arg);
    }
  }

  std::vector<const char*> argv()
  {
    argv_.clear();
    argv_.reserve(args_.size());
    for (const auto& arg : args_)
    {
      argv_.push_back(arg.is_raw ? arg.raw : arg.value.c_str());
    }
    return argv_;
  }

private:
  struct Arg
  {
    std::string value;
    const char* raw = nullptr;
    bool is_raw     = false;
  };

  std::vector<Arg> args_;
  std::vector<const char*> argv_;
};

inline std::string get_cudacc_output_log(const cudaccOutput& output)
{
  if (!output.program_log)
  {
    return {};
  }
  return std::string(output.program_log, output.program_log_size);
}
} // namespace hostjit::detail
