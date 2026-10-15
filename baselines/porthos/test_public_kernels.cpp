#include "causal_round_counter.h"
#include "ghostmpc_public_kernels.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

int main() {
  using namespace ghostmpc_porthos;
  const std::uint64_t a[] = {std::uint64_t(-2), 3, 4, 5};
  const std::uint64_t b[] = {7, std::uint64_t(-1), 2, 6};
  std::uint64_t c[4]{};
  matmul_public(2, 2, 2, a, b, c);
  assert(c[0] == std::uint64_t(-8) && c[1] == 20);
  assert(c[2] == 38 && c[3] == 26);

  const std::uint64_t wrap[] = {std::numeric_limits<std::uint64_t>::max()};
  const std::uint64_t two[] = {2};
  std::uint64_t product[1]{};
  mul_public(1, wrap, two, product);
  assert(product[0] == std::numeric_limits<std::uint64_t>::max() - 1);

  std::uint64_t left[1]{}, right[1]{};
  const std::uint64_t share[] = {7}, constant[] = {9};
  add_public(1, share, constant, left, true);
  add_public(1, share, constant, right, false);
  assert(left[0] + right[0] == 23);  // 7+7+9, constant exactly once.

  std::vector<std::uint64_t> input(3 * 3 * 2);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = i + 1;
  std::vector<std::uint64_t> filter(3 * 3 * 1 * 4, 1);
  std::vector<std::uint64_t> output(3 * 3 * 4);
  conv2d_public(1, 3, 3, 2, 3, 3, 4, 1, 1, 1, 1, 1, 1, 2,
                input.data(), filter.data(), output.data());
  std::uint64_t even = 0, odd = 0;
  for (std::size_t i = 0; i < input.size(); i += 2) {
    even += input[i]; odd += input[i + 1];
  }
  assert(output[(1 * 3 + 1) * 4] == even);
  assert(output[(1 * 3 + 1) * 4 + 2] == odd);

  std::vector<std::uint64_t> wide_input(4 * 4 * 65, 1);
  std::vector<std::uint64_t> wide_filter(65 * 65, 1);
  std::vector<std::uint64_t> wide_output(2 * 2 * 65);
  conv2d_public(1, 4, 4, 65, 1, 1, 65, 0, 0, 0, 0, 2, 2, 1,
                wide_input.data(), wide_filter.data(), wide_output.data());
  for (const auto value : wide_output) assert(value == 65);

  CausalRoundCounter rounds;
  rounds.event("parallel-sends", false);
  rounds.event("receive-1", true, {"parallel-sends"});
  rounds.event("local", false, {"receive-1"});
  rounds.event("receive-2", true, {"local"});
  assert(rounds.rounds() == 2);
}
