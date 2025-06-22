#pragma once
#include <sys/resource.h>
#include <unistd.h>

// Optional logging support
#ifdef USE_GLIB_LOGGING
#include <glib.h>
#define FR_DEBUG(fmt, ...) g_debug(fmt, ##__VA_ARGS__)
#define FR_INFO(fmt, ...) g_info(fmt, ##__VA_ARGS__)
#define FR_WARNING(fmt, ...) g_warning(fmt, ##__VA_ARGS__)
#define FR_ERROR(fmt, ...) g_error(fmt, ##__VA_ARGS__)
#else
#define FR_DEBUG(fmt, ...)                                                                         \
  do {                                                                                             \
    std::printf("\033[34m [DEBUG] " fmt "\033[0m\n", ##__VA_ARGS__);                            \
  } while (0)
#define FR_INFO(fmt, ...)                                                                          \
  do {                                                                                             \
    std::printf("\033[32m󰋼 [INFO] " fmt "\033[0m\n", ##__VA_ARGS__);                            \
  } while (0)
#define FR_WARNING(fmt, ...)                                                                       \
  do {                                                                                             \
    std::printf("\033[33m⚠️ [WARNING] " fmt "\033[0m\n", ##__VA_ARGS__);                            \
  } while (0)
#define FR_ERROR(fmt, ...)                                                                         \
  do {                                                                                             \
    std::printf("\033[31m❌ [ERROR] " fmt "\033[0m\n", ##__VA_ARGS__);                             \
    std::exit(1);                                                                                  \
  } while (0)
#endif

inline void disableCoreDumps() {
  struct rlimit core_limits;
  core_limits.rlim_cur = 0;
  core_limits.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &core_limits);
}
