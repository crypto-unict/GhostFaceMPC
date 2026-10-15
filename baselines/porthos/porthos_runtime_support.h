#ifndef GHOSTMPC_PORTHOS_RUNTIME_SUPPORT_H
#define GHOSTMPC_PORTHOS_RUNTIME_SUPPORT_H

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

extern int partyNum;

namespace ghostmpc_porthos {

inline int request_count() {
  const char* raw = std::getenv("GHOSTMPC_REQUESTS");
  if (!raw) return 1;
  const int value = std::stoi(raw);
  if (value < 1) throw std::invalid_argument("GHOSTMPC_REQUESTS must be positive");
  return value;
}

inline void write_share(std::uint64_t value, int request) {
  if (partyNum == 2) return;
  const char* path = std::getenv("GHOSTMPC_OUTPUT_SHARE");
  if (!path) throw std::runtime_error("GHOSTMPC_OUTPUT_SHARE is not set");
  static std::ofstream stream(path, std::ios::out | std::ios::trunc);
  if (!stream.good()) throw std::runtime_error("cannot open output share file");
  stream << request << ' ' << value << '\n';
}

}  // namespace ghostmpc_porthos

#endif
