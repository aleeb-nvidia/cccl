#pragma once

#include <string>
#include <system_error>

namespace hostjit
{
class DynamicLibrary
{
public:
  DynamicLibrary();
  ~DynamicLibrary();

  // Disable copy
  DynamicLibrary(const DynamicLibrary&)            = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  // Enable move
  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  // Load a shared library
  bool load(const std::string& library_path);

  // Get a symbol (function or variable) by name
  void* getSymbol(const std::string& symbol_name);

  // Template helper to get exported symbols with type safety
  template <typename T>
  T getSymbolAs(const std::string& name)
  {
    return reinterpret_cast<T>(getSymbol(name));
  }

  // Check if library is loaded
  bool isLoaded() const;

  // Get the last error message
  std::string getLastError() const;

  // Get the last error code
  std::error_code getLastErrorCode() const;

  // Unload the library
  void unload();

private:
  void* handle_;
  std::string last_error_;
  std::error_code last_error_code_;
};
} // namespace hostjit
