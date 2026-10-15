#ifndef GHOSTMPC_PORTHOS_CAUSAL_ROUND_COUNTER_H
#define GHOSTMPC_PORTHOS_CAUSAL_ROUND_COUNTER_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ghostmpc_porthos {

// Primitive implementations mark a receive-dependent phase once, regardless
// of how many independent sends execute in parallel in that phase.
class CausalRoundCounter {
 public:
  void reset() { depth_.clear(); maximum_ = 0; }

  std::size_t event(const std::string& id, bool receive_dependent,
                    const std::vector<std::string>& dependencies = {}) {
    if (depth_.count(id)) throw std::invalid_argument("duplicate causal event");
    std::size_t prior = 0;
    for (const auto& dependency : dependencies) {
      const auto found = depth_.find(dependency);
      if (found == depth_.end())
        throw std::invalid_argument("missing causal dependency");
      prior = std::max(prior, found->second);
    }
    const std::size_t current = prior + (receive_dependent ? 1 : 0);
    depth_[id] = current;
    maximum_ = std::max(maximum_, current);
    return current;
  }

  std::size_t rounds() const { return maximum_; }

 private:
  std::unordered_map<std::string, std::size_t> depth_;
  std::size_t maximum_ = 0;
};

// The runtime realization of the same rule, driven by the socket layer rather
// than by a declared event graph.  A round opens on the first receive that
// follows a non-receive; consecutive receives, from any peer, belong to the
// same round, which is what makes independent parallel receives one round.
//
// `begin_exchange`/`end_exchange` exist because Porthos runs a send and a
// receive on two threads at its PARALLEL_COMM sites and inside `synchronize`:
// no local order exists between them, so the socket-level rule would count
// such a site zero or one times depending on the scheduler.  Each of those
// sites is one simultaneous exchange, and it announces itself *before* the
// spawn -- which happens-before every socket call the threads then make -- so
// the count does not depend on the interleaving.  An announced exchange ends
// having received, hence the flag it leaves behind.
class SerialRoundCounter {
 public:
  void reset() {
    rounds_.store(0, std::memory_order_relaxed);
    last_was_receive_.store(false, std::memory_order_relaxed);
    phase_.store(0, std::memory_order_relaxed);
    armed_.store(false, std::memory_order_relaxed);
  }

  void arm(bool value) { armed_.store(value, std::memory_order_relaxed); }

  bool armed() const { return armed_.load(std::memory_order_relaxed); }

  void sent() {
    if (!armed() || phase_.load(std::memory_order_acquire) != 0) return;
    last_was_receive_.store(false, std::memory_order_relaxed);
  }

  void received() {
    if (!armed() || phase_.load(std::memory_order_acquire) != 0) return;
    if (!last_was_receive_.exchange(true, std::memory_order_relaxed))
      rounds_.fetch_add(1, std::memory_order_relaxed);
  }

  void begin_exchange() {
    if (armed() && phase_.load(std::memory_order_relaxed) == 0) {
      rounds_.fetch_add(1, std::memory_order_relaxed);
      last_was_receive_.store(true, std::memory_order_relaxed);
    }
    phase_.fetch_add(1, std::memory_order_release);
  }

  void end_exchange() { phase_.fetch_sub(1, std::memory_order_release); }

  std::size_t rounds() const { return rounds_.load(std::memory_order_relaxed); }

 private:
  std::atomic<std::size_t> rounds_{0};
  std::atomic<bool> last_was_receive_{false};
  std::atomic<int> phase_{0};
  std::atomic<bool> armed_{false};
};

}  // namespace ghostmpc_porthos

#endif
