#include "EzPCFunctionalities.h"
#include "ghostmpc_public_kernels.h"

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <vector>

extern int partyNum;
extern CommunicationObject commObject;

namespace {
template <class T>
std::uint64_t ring(T value) {
  return static_cast<std::uint64_t>(value);
}

// The phases outside the online window.  Porthos accumulates bytes
// and rounds only while `measurement` is armed, and `start_communication()`
// refuses to nest, so a phase measured outside `start_m()`/`end_m()` arms the
// counters itself and disarms them before it returns.  Nothing here changes
// what the online window contains: it reports what was already excluded.
struct GhostPhase {
  timespec wall;
  timespec cpu;
};

GhostPhase ghostmpc_offline_phase;
GhostPhase ghostmpc_sync_phase;

double ghostmpc_elapsed(const timespec& from, const timespec& to) {
  return (to.tv_sec - from.tv_sec) +
         (to.tv_nsec - from.tv_nsec) / 1000000000.0;
}

void ghostmpc_phase_begin(GhostPhase& phase) {
  commObject.reset();
  commObject.setMeasurement(true);
  clock_gettime(CLOCK_MONOTONIC, &phase.wall);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &phase.cpu);
}

void ghostmpc_phase_end(const char* name, const GhostPhase& phase) {
  timespec wall;
  timespec cpu;
  clock_gettime(CLOCK_MONOTONIC, &wall);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu);
  const std::size_t sent = commObject.getSent();
  const std::size_t received = commObject.getRecv();
  const std::size_t rounds = commObject.getRounds();
  const std::size_t sends = commObject.getRoundsSent();
  const std::size_t receives = commObject.getRoundsRecv();
  commObject.setMeasurement(false);
  commObject.reset();
  const std::streamsize precision = std::cout.precision(12);
  std::cout << name << " {\"party\":" << partyNum
            << ",\"wall_seconds\":" << ghostmpc_elapsed(phase.wall, wall)
            << ",\"cpu_seconds\":" << ghostmpc_elapsed(phase.cpu, cpu)
            << ",\"bytes_sent\":" << sent
            << ",\"bytes_received\":" << received
            << ",\"rounds\":" << rounds
            << ",\"send_calls\":" << sends
            << ",\"recv_calls\":" << receives
            << "}" << std::endl;
  std::cout.precision(precision);
}
}

// Setup: session keys, the mod-prime tables, the sockets, the one setup
// barrier, the public weight stream and the enrolled gallery row.  Porthos
// exchanges no correlated randomness -- its "offline module" populates local
// tables and its PRG keys are pre-shared files -- so the only offline traffic
// is the connection setup and that barrier.
void GhostOfflineBegin() { ghostmpc_phase_begin(ghostmpc_offline_phase); }

void GhostOfflineEnd() {
  ghostmpc_phase_end("GHOSTMPC_PORTHOS_OFFLINE", ghostmpc_offline_phase);
}

// The barrier `StartComputation()` runs before `start_m()`, once per request.
// It is outside the online figure, and is reported separately.
void GhostSyncBegin() { ghostmpc_phase_begin(ghostmpc_sync_phase); }

void GhostSyncEnd() {
  ghostmpc_phase_end("GHOSTMPC_PORTHOS_SYNC", ghostmpc_sync_phase);
}

bool GhostPorthosNearestEnabled() {
  const char* value = std::getenv("GHOSTMPC_PORTHOS_NEAREST");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

void GhostTruncateNearest(std::vector<porthosSecretType>& values,
                          std::size_t power, std::size_t size) {
  if (!GhostPorthosNearestEnabled())
    throw std::logic_error("secure nearest truncation was not enabled");
  if (power == 0 || power >= 63 || values.size() < size)
    throw std::invalid_argument("secure nearest truncation requires 0 < power < 63");

  // Rounding half up is floor((x + 2^(power-1)) / 2^power), matching
  // MP-SPDZ TruncRoundNearest. Only P0 adds the public half ULP.
  std::vector<porthosSecretType> rounded(values);
  if (partyNum == PARTY_A) {
    const porthosSecretType half = porthosSecretType(1) << (power - 1);
    for (std::size_t i = 0; i < size; ++i) rounded[i] += half;
  }

  // Exact signed shifting needs the carry into bit `power` and the sign of
  // A+B. Both comparisons are performed by Porthos' three-party MSB protocol
  // and return raw (unscaled) bit shares in nearest mode. Comparing the carry
  // into bit 63 directly would span the entire signed ring, outside the
  // doubled-input bound of funcRELUPrime3PC; comparing the bounded truncand
  // itself avoids that failure. No primary reconstructs A+B or either bit.
  std::vector<porthosSecretType> comparisons(2 * size, 0);
  if (PRIMARY) {
    const porthosSecretType low_mask =
        (porthosSecretType(1) << power) - 1;
    for (std::size_t i = 0; i < size; ++i) {
      comparisons[i] = rounded[i] & low_mask;
      comparisons[size + i] = rounded[i];
      if (partyNum == PARTY_A) {
        comparisons[i] -= porthosSecretType(1) << power;
      }
    }
  }
  std::vector<porthosSecretType> carries(2 * size, 0);
  funcRELUPrime3PC(comparisons, carries, 2 * size);

  // Let alpha and beta be the top bits of A and B, and sign the result MSB.
  // The correction (carry_out + sign) * 2^(64-power) is the following
  // multilinear polynomial over those secret bits:
  // alpha+beta+sign-alpha*beta-alpha*sign-beta*sign+2*alpha*beta*sign.
  // The products below use Beaver multiplication without fixed-point
  // truncation because every operand is a raw secret bit.
  std::vector<porthosSecretType> alpha(size, 0), beta(size, 0);
  if (partyNum == PARTY_A)
    for (std::size_t i = 0; i < size; ++i) alpha[i] = rounded[i] >> 63;
  if (partyNum == PARTY_B)
    for (std::size_t i = 0; i < size; ++i) beta[i] = rounded[i] >> 63;
  std::vector<porthosSecretType> sign(size, 0);
  if (PRIMARY) {
    for (std::size_t i = 0; i < size; ++i)
      sign[i] = (partyNum == PARTY_A ? 1 : 0) - carries[size + i];
  }

  std::vector<porthosSecretType> lhs(3 * size, 0), rhs(3 * size, 0);
  if (PRIMARY) {
    for (std::size_t i = 0; i < size; ++i) {
      lhs[i] = alpha[i];
      rhs[i] = beta[i];
      lhs[size + i] = alpha[i];
      rhs[size + i] = sign[i];
      lhs[2 * size + i] = beta[i];
      rhs[2 * size + i] = sign[i];
    }
  }
  std::vector<porthosSecretType> pair_products(3 * size, 0);
  funcDotProductMPC(lhs, rhs, pair_products, 3 * size, 0, false);

  std::vector<porthosSecretType> alpha_beta(size, 0), sign_copy(size, 0),
      triple(size, 0);
  if (PRIMARY) {
    for (std::size_t i = 0; i < size; ++i) {
      alpha_beta[i] = pair_products[i];
      sign_copy[i] = sign[i];
    }
  }
  funcDotProductMPC(alpha_beta, sign_copy, triple, size, 0, false);

  if (PRIMARY) {
    const porthosSecretType signed_unit =
        porthosSecretType(1) << (64 - power);
    for (std::size_t i = 0; i < size; ++i) {
      const porthosSecretType correction_bits =
          alpha[i] + beta[i] + sign[i] - pair_products[i] -
          pair_products[size + i] - pair_products[2 * size + i] +
          2 * triple[i];
      values[i] = (rounded[i] >> power) + carries[i] -
                  correction_bits * signed_unit;
    }
  } else {
    std::fill(values.begin(), values.begin() + size, 0);
  }
}

namespace {
void scale_public(int64_t& value, int32_t sf) {
  value = int64_t(ring(value) << sf);
}

template <class T>
void scale_public(std::vector<T>& values, int32_t sf) {
  for (auto& value : values) scale_public(value, sf);
}
}

porthosSecretType GhostAddPublicScalar(porthosSecretType secret, int64_t clear) {
  return secret + (partyNum == PARTY_A ? ring(clear) : 0);
}

porthosSecretType GhostSecretMinusPublicScalar(porthosSecretType secret,
                                                int64_t clear) {
  return secret - (partyNum == PARTY_A ? ring(clear) : 0);
}

porthosSecretType GhostPublicMinusSecretScalar(porthosSecretType secret,
                                                int64_t clear) {
  return (partyNum == PARTY_A ? ring(clear) : 0) - secret;
}

void GhostMatMulPublic(int32_t rows, int32_t inner, int32_t columns,
                       std::vector<std::vector<porthosSecretType>>& share,
                       std::vector<std::vector<int64_t>>& weight,
                       std::vector<std::vector<porthosSecretType>>& output,
                       bool) {
  std::vector<std::uint64_t> a(rows * inner), b(inner * columns),
      c(rows * columns);
  for (int32_t i = 0; i < rows; ++i)
    for (int32_t k = 0; k < inner; ++k) a[i * inner + k] = share[i][k];
  for (int32_t k = 0; k < inner; ++k)
    for (int32_t j = 0; j < columns; ++j) b[k * columns + j] = ring(weight[k][j]);
  ghostmpc_porthos::matmul_public(rows, inner, columns, a.data(), b.data(), c.data());
  for (int32_t i = 0; i < rows; ++i)
    for (int32_t j = 0; j < columns; ++j) output[i][j] = c[i * columns + j];
}

void GhostElemWiseMulPublic(int32_t size,
                            std::vector<porthosSecretType>& share,
                            std::vector<int64_t>& weight,
                            std::vector<porthosSecretType>& output) {
  ghostmpc_porthos::mul_public(
      size, share.data(), reinterpret_cast<const std::uint64_t*>(weight.data()),
      output.data());
}

void GhostScaleUpPublic1(int32_t s1, std::vector<int64_t>& values, int32_t sf) {
  (void)s1;
  scale_public(values, sf);
}

void GhostScaleUpPublic2(int32_t, int32_t,
                         std::vector<std::vector<int64_t>>& values,
                         int32_t sf) {
  scale_public(values, sf);
}

void GhostScaleUpPublic3(
    int32_t, int32_t, int32_t,
    std::vector<std::vector<std::vector<int64_t>>>& values, int32_t sf) {
  scale_public(values, sf);
}

void GhostScaleUpPublic4(
    int32_t, int32_t, int32_t, int32_t,
    std::vector<std::vector<std::vector<std::vector<int64_t>>>>& values,
    int32_t sf) {
  scale_public(values, sf);
}

void GhostScaleUpPublic5(
    int32_t, int32_t, int32_t, int32_t, int32_t,
    std::vector<std::vector<std::vector<std::vector<std::vector<int64_t>>>>>&
        values,
    int32_t sf) {
  scale_public(values, sf);
}

void HardSigmoid(int32_t, std::vector<porthosSecretType>&,
                 std::vector<porthosSecretType>&, int32_t, bool) {
  // Library64_common.ezpc always emits HardSigmoid wrappers. GhostMPC lowers
  // every model HardSigmoid to public affine/ReLU primitives before Athos, so
  // this compatibility symbol must link but must never execute silently.
  throw std::logic_error("HardSigmoid must be lowered before Porthos codegen");
}

void ClearMemPublic1(int64_t, std::vector<int64_t>& value) { value = {}; }
void ClearMemPublic2(int64_t, int64_t,
                     std::vector<std::vector<int64_t>>& value) { value = {}; }
void ClearMemPublic3(int64_t, int64_t, int64_t,
                     std::vector<std::vector<std::vector<int64_t>>>& value) {
  value = {};
}
void ClearMemPublic4(
    int64_t, int64_t, int64_t, int64_t,
    std::vector<std::vector<std::vector<std::vector<int64_t>>>>& value) {
  value = {};
}
void ClearMemPublic5(
    int64_t, int64_t, int64_t, int64_t, int64_t,
    std::vector<std::vector<std::vector<std::vector<std::vector<int64_t>>>>>& value) {
  value = {};
}

void GhostConv2DGroupPublicWrapper(
    int32_t n, int32_t h, int32_t w, int32_t ci, int32_t fh, int32_t fw,
    int32_t co, int32_t pad_top, int32_t pad_bottom, int32_t pad_left,
    int32_t pad_right, int32_t stride_h, int32_t stride_w, int32_t group,
    std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>& input,
    std::vector<std::vector<std::vector<std::vector<int64_t>>>>& filter,
    std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>& output) {
  const int32_t oh = (h + pad_top + pad_bottom - fh) / stride_h + 1;
  const int32_t ow = (w + pad_left + pad_right - fw) / stride_w + 1;
  std::vector<std::uint64_t> flat_input(std::size_t(n) * h * w * ci);
  std::vector<std::uint64_t> flat_filter(std::size_t(fh) * fw * (ci / group) * co);
  std::vector<std::uint64_t> flat_output(std::size_t(n) * oh * ow * co);
  for (int32_t b = 0; b < n; ++b)
    for (int32_t y = 0; y < h; ++y)
      for (int32_t x = 0; x < w; ++x)
        for (int32_t c = 0; c < ci; ++c)
          flat_input[((b * h + y) * w + x) * ci + c] = input[b][y][x][c];
  for (int32_t y = 0; y < fh; ++y)
    for (int32_t x = 0; x < fw; ++x)
      for (int32_t c = 0; c < ci / group; ++c)
        for (int32_t o = 0; o < co; ++o)
          flat_filter[((y * fw + x) * (ci / group) + c) * co + o] =
              ring(filter[y][x][c][o]);
  ghostmpc_porthos::conv2d_public(
      n, h, w, ci, fh, fw, co, pad_top, pad_bottom, pad_left, pad_right,
      stride_h, stride_w, group, flat_input.data(), flat_filter.data(),
      flat_output.data());
  for (int32_t b = 0; b < n; ++b)
    for (int32_t y = 0; y < oh; ++y)
      for (int32_t x = 0; x < ow; ++x)
        for (int32_t c = 0; c < co; ++c)
          output[b][y][x][c] = flat_output[((b * oh + y) * ow + x) * co + c];
}

void GhostConv2DPublicWrapper(
    int32_t n, int32_t h, int32_t w, int32_t ci, int32_t fh, int32_t fw,
    int32_t co, int32_t pad_top, int32_t pad_bottom, int32_t pad_left,
    int32_t pad_right, int32_t stride_h, int32_t stride_w,
    std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>& input,
    std::vector<std::vector<std::vector<std::vector<int64_t>>>>& filter,
    std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>& output) {
  GhostConv2DGroupPublicWrapper(
      n, h, w, ci, fh, fw, co, pad_top, pad_bottom, pad_left, pad_right,
      stride_h, stride_w, 1, input, filter, output);
}

// The 1:1 verification score, in Porthos' own arithmetic.
//
// Porthos has a genuinely secret-by-secret product --
// `ElemWiseSecretSharedVectorMult` reduces to `funcDotProductMPC` on two shared
// vectors.  The score is nevertheless built here rather than in the ONNX graph:
// the assertion that blocks a product of two secrets lives in Athos' shared
// SeeDot IR builder (`IRBuilderCSF.visitBopMul2DTensor`, "expecting one of A or
// B to be an input by the server"), so it applies to every Athos target.
// Building it from Porthos' own primitives prices what Porthos charges rather
// than a hand-written kernel.
//
// The truncation is deliberately `ScaleDown` and not a nearest-rounding call:
// `ScaleDown` dispatches to `GhostTruncateNearest` exactly when the nearest
// truncation is selected, so the score truncates the way every other layer of
// the same binary truncates.
//
// Nothing is opened: each party keeps its additive share of the two results.
void GhostScore(int32_t size, std::vector<porthosSecretType>& embedding,
                std::vector<porthosSecretType>& gallery, int32_t sf,
                std::vector<porthosSecretType>& out) {
  std::vector<porthosSecretType> products(size);
  std::vector<porthosSecretType> squares(size);
  // The squares need the probe as both operands and the primitive takes
  // mutable references, so the second operand is a copy, not an alias.
  std::vector<porthosSecretType> probe(embedding);
  ElemWiseSecretSharedVectorMult(size, embedding, gallery, products);
  ElemWiseSecretSharedVectorMult(size, embedding, probe, squares);
  ScaleDown(size, products, sf);
  ScaleDown(size, squares, sf);
  // Additive shares add locally: no communication, and nothing reconstructed.
  porthosSecretType similarity = 0;
  porthosSecretType norm = 0;
  for (int32_t i = 0; i < size; ++i) {
    similarity += products[i];
    norm += squares[i];
  }
  out.assign({similarity, norm});
}
