#ifndef GHOSTMPC_PORTHOS_PUBLIC_KERNELS_H
#define GHOSTMPC_PORTHOS_PUBLIC_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ghostmpc_porthos {

// Unsigned arithmetic gives defined wrap in Z/(2^64). These kernels never
// reference PorthosNet or CommunicationObject and therefore send zero bytes.
inline void matmul_public(std::size_t rows, std::size_t inner,
                          std::size_t cols, const std::uint64_t* share,
                          const std::uint64_t* weight,
                          std::uint64_t* output) {
  if (!share || !weight || !output) throw std::invalid_argument("null matrix");
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      std::uint64_t accumulator = 0;
      for (std::size_t k = 0; k < inner; ++k)
        accumulator += share[row * inner + k] * weight[k * cols + col];
      output[row * cols + col] = accumulator;
    }
  }
}

inline void mul_public(std::size_t size, const std::uint64_t* share,
                       const std::uint64_t* weight,
                       std::uint64_t* output) {
  if (!share || !weight || !output) throw std::invalid_argument("null vector");
  for (std::size_t index = 0; index < size; ++index)
    output[index] = share[index] * weight[index];
}

inline void add_public(std::size_t size, const std::uint64_t* share,
                       const std::uint64_t* value, std::uint64_t* output,
                       bool constant_owner) {
  if (!share || !value || !output) throw std::invalid_argument("null vector");
  for (std::size_t index = 0; index < size; ++index)
    output[index] = share[index] + (constant_owner ? value[index] : 0);
}

inline void secret_minus_public(
    std::size_t size, const std::uint64_t* share,
    const std::uint64_t* value, std::uint64_t* output,
    bool constant_owner) {
  if (!share || !value || !output) throw std::invalid_argument("null vector");
  for (std::size_t index = 0; index < size; ++index)
    output[index] = share[index] - (constant_owner ? value[index] : 0);
}

inline void public_minus_secret(
    std::size_t size, const std::uint64_t* value,
    const std::uint64_t* share, std::uint64_t* output,
    bool constant_owner) {
  if (!share || !value || !output) throw std::invalid_argument("null vector");
  for (std::size_t index = 0; index < size; ++index)
    output[index] = (constant_owner ? value[index] : 0) - share[index];
}

// NHWC input, [FH,FW,CI/group,CO] filter, NHWC output. Depthwise with a
// multiplier greater than one is group==CI and CO/CI==multiplier.
inline void conv2d_public(
    std::size_t n, std::size_t h, std::size_t w, std::size_t ci,
    std::size_t fh, std::size_t fw, std::size_t co,
    std::size_t pad_top, std::size_t pad_bottom,
    std::size_t pad_left, std::size_t pad_right,
    std::size_t stride_h, std::size_t stride_w, std::size_t group,
    const std::uint64_t* share, const std::uint64_t* filter,
    std::uint64_t* output) {
  if (!share || !filter || !output) throw std::invalid_argument("null tensor");
  if (!group || ci % group || co % group || !stride_h || !stride_w)
    throw std::invalid_argument("invalid grouped convolution dimensions");
  if (h + pad_top + pad_bottom < fh || w + pad_left + pad_right < fw)
    throw std::invalid_argument("kernel exceeds padded input");
  const std::size_t oh = (h + pad_top + pad_bottom - fh) / stride_h + 1;
  const std::size_t ow = (w + pad_left + pad_right - fw) / stride_w + 1;
  const std::size_t ci_group = ci / group;
  const std::size_t co_group = co / group;
  for (std::size_t batch = 0; batch < n; ++batch)
    for (std::size_t oy = 0; oy < oh; ++oy)
      for (std::size_t ox = 0; ox < ow; ++ox)
        for (std::size_t oc = 0; oc < co; ++oc) {
          std::uint64_t accumulator = 0;
          const std::size_t current_group = oc / co_group;
          for (std::size_t ky = 0; ky < fh; ++ky) {
            const std::ptrdiff_t iy = static_cast<std::ptrdiff_t>(
                oy * stride_h + ky) - static_cast<std::ptrdiff_t>(pad_top);
            if (iy < 0 || iy >= static_cast<std::ptrdiff_t>(h)) continue;
            for (std::size_t kx = 0; kx < fw; ++kx) {
              const std::ptrdiff_t ix = static_cast<std::ptrdiff_t>(
                  ox * stride_w + kx) - static_cast<std::ptrdiff_t>(pad_left);
              if (ix < 0 || ix >= static_cast<std::ptrdiff_t>(w)) continue;
              for (std::size_t kc = 0; kc < ci_group; ++kc) {
                const std::size_t ic = current_group * ci_group + kc;
                const std::size_t input_index =
                    ((batch * h + static_cast<std::size_t>(iy)) * w +
                     static_cast<std::size_t>(ix)) * ci + ic;
                const std::size_t filter_index =
                    ((ky * fw + kx) * ci_group + kc) * co + oc;
                accumulator += share[input_index] * filter[filter_index];
              }
            }
          }
          output[((batch * oh + oy) * ow + ox) * co + oc] = accumulator;
        }
}

}  // namespace ghostmpc_porthos

#endif
