#include <hostjit/loader.hpp>

#include <cstring>
#include <utility>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace hostjit
{
namespace
{
enum class dynamic_library_errc
{
  not_loaded = 1,
  load_failed,
  symbol_not_found
};

class DynamicLibraryErrorCategory : public std::error_category
{
public:
  const char* name() const noexcept override
  {
    return "hostjit.dynamic_library";
  }

  std::string message(int ev) const override
  {
    switch (static_cast<dynamic_library_errc>(ev))
    {
      case dynamic_library_errc::not_loaded:
        return "library not loaded";
      case dynamic_library_errc::load_failed:
        return "library load failed";
      case dynamic_library_errc::symbol_not_found:
        return "symbol lookup failed";
    }
    return "unknown dynamic library error";
  }
};

const std::error_category& dynamic_library_category()
{
  static const DynamicLibraryErrorCategory category;
  return category;
}

std::error_code make_error_code(dynamic_library_errc errc)
{
  return {static_cast<int>(errc), dynamic_library_category()};
}

#ifdef _WIN32
// Run C++ static constructors in a DLL loaded with /NOENTRY /NODEFAULTLIB.
// The compiler places CUDA fatbin registration in the .CRT$XCU section.
// Without CRT startup, these never run, so we walk the merged .CRT section
// in the PE and call each non-null function pointer.
void runStaticInitializers(HMODULE module)
{
  auto base = reinterpret_cast<const unsigned char*>(module);
  auto dos  = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  auto nt   = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
  auto sec  = IMAGE_FIRST_SECTION(nt);

  for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec)
  {
    if (memcmp(sec->Name, ".CRT", 4) == 0)
    {
      using InitFunc = void(__cdecl*)();
      auto funcs     = reinterpret_cast<InitFunc*>(const_cast<unsigned char*>(base) + sec->VirtualAddress);
      size_t count   = sec->SizeOfRawData / sizeof(InitFunc);
      for (size_t j = 0; j < count; ++j)
      {
        if (funcs[j])
        {
          funcs[j]();
        }
      }
    }
  }
}
#endif
} // anonymous namespace

DynamicLibrary::DynamicLibrary()
    : handle_(nullptr)
{}

DynamicLibrary::~DynamicLibrary()
{
  unload();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(other.handle_)
    , last_error_(std::move(other.last_error_))
    , last_error_code_(std::move(other.last_error_code_))
{
  other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept
{
  if (this != &other)
  {
    unload();
    handle_           = other.handle_;
    last_error_       = std::move(other.last_error_);
    last_error_code_  = std::move(other.last_error_code_);
    other.handle_     = nullptr;
  }
  return *this;
}

bool DynamicLibrary::load(const std::string& library_path)
{
  unload();

#ifdef _WIN32
  SetLastError(0);
  handle_ = static_cast<void*>(LoadLibraryA(library_path.c_str()));

  if (!handle_)
  {
    DWORD error = GetLastError();
    if (error != 0)
    {
      last_error_code_ = std::error_code(static_cast<int>(error), std::system_category());
      last_error_      = last_error_code_.message();
    }
    else
    {
      last_error_ = "Unknown LoadLibrary error";
      last_error_code_ = make_error_code(dynamic_library_errc::load_failed);
    }
    return false;
  }

  // The DLL is linked with /NOENTRY (no CRT startup), so C++ static
  // constructors (e.g. CUDA fatbin registration) haven't run yet.
  runStaticInitializers(static_cast<HMODULE>(handle_));
#else
  dlerror();
  handle_ = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (!handle_)
  {
    const char* error = dlerror();
    last_error_       = error ? error : "Unknown dlopen error";
    last_error_code_  = make_error_code(dynamic_library_errc::load_failed);
    return false;
  }
#endif

  last_error_.clear();
  last_error_code_.clear();
  return true;
}

void* DynamicLibrary::getSymbol(const std::string& symbol_name)
{
  if (!handle_)
  {
    last_error_      = "Library not loaded";
    last_error_code_ = make_error_code(dynamic_library_errc::not_loaded);
    return nullptr;
  }

#ifdef _WIN32
  SetLastError(0);
  void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), symbol_name.c_str()));

  if (!symbol)
  {
    DWORD error = GetLastError();
    if (error != 0)
    {
      last_error_code_ = std::error_code(static_cast<int>(error), std::system_category());
      last_error_      = last_error_code_.message();
    }
    else
    {
      last_error_ = "Symbol not found: " + symbol_name;
      last_error_code_ = make_error_code(dynamic_library_errc::symbol_not_found);
    }
    return nullptr;
  }
#else
  dlerror();
  void* symbol = dlsym(handle_, symbol_name.c_str());

  const char* error = dlerror();
  if (error)
  {
    last_error_      = error;
    last_error_code_ = make_error_code(dynamic_library_errc::symbol_not_found);
    return nullptr;
  }
#endif

  last_error_.clear();
  last_error_code_.clear();
  return symbol;
}

bool DynamicLibrary::isLoaded() const
{
  return handle_ != nullptr;
}

std::string DynamicLibrary::getLastError() const
{
  return last_error_;
}

std::error_code DynamicLibrary::getLastErrorCode() const
{
  return last_error_code_;
}

void DynamicLibrary::unload()
{
  if (handle_)
  {
    // Intentionally do NOT unload (dlclose / FreeLibrary) a compiled JIT module. See #9367.
    //
    // Each JIT .so is built by Clang with the classic fatbin embedding (-fcuda-include-gpubinary),
    // which emits a module ctor (__cuda_module_ctor -> __cudaRegisterFatBinary)
    // in .init_array but NO matching unregister dtor (.fini_array / __cudaUnregisterFatBinary).
    // Unloading such a module unmaps its fatbin while the CUDA runtime still holds a pointer to it;
    // that dangling registration corrupts the runtime's module table, so a later module's kernel
    // launch silently no-ops.
    handle_ = nullptr;
  }
  last_error_.clear();
  last_error_code_.clear();
}
} // namespace hostjit
