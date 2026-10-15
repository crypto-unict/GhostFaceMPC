// Build gate for the round counter: the runtime round counter the
// measured binary runs, driven through microprotocols whose round complexity
// is known, and cross-checked against the declarative causal DAG.
#include "causal_round_counter.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

using ghostmpc_porthos::CausalRoundCounter;
using ghostmpc_porthos::SerialRoundCounter;

namespace {

void reset_armed(SerialRoundCounter& counter) {
  counter.reset();
  counter.arm(true);
}

// funcReconstruct2PC as the primaries run it: one party sends, the other
// receives, and the phase is one round on both sides.
std::size_t reconstruct(SerialRoundCounter& counter, bool sender) {
  if (sender) counter.sent(); else counter.received();
  return counter.rounds();
}

}  // namespace

int main() {
  SerialRoundCounter counter;

  // 1. A single reconstruct is one round for the receiver and zero for the
  //    sender: a send is not a round, the party that waits pays it.
  reset_armed(counter);
  assert(reconstruct(counter, true) == 0);
  reset_armed(counter);
  assert(reconstruct(counter, false) == 1);

  // 2. k dependent reconstructs are k rounds.  This is the shape of a chain of
  //    truncations: send, receive, send, receive, ...
  for (std::size_t k = 1; k <= 8; ++k) {
    reset_armed(counter);
    CausalRoundCounter reference;
    for (std::size_t i = 0; i < k; ++i) {
      counter.sent();
      counter.received();
      const std::string previous = i ? "recv" + std::to_string(i - 1) : "";
      reference.event("send" + std::to_string(i), false,
                      previous.empty() ? std::vector<std::string>{}
                                       : std::vector<std::string>{previous});
      reference.event("recv" + std::to_string(i), true,
                      {"send" + std::to_string(i)});
    }
    assert(counter.rounds() == k);
    assert(counter.rounds() == reference.rounds());
  }

  // 3. Independent parallel receives are one round.  funcShareConvertMPC has
  //    P2 receive a_tilde from both primaries in one phase, and each primary
  //    receive the bit and delta shares from P2 in one phase.
  reset_armed(counter);
  counter.received();
  counter.received();
  {
    CausalRoundCounter reference;
    reference.event("from-a", true);
    reference.event("from-b", true);
    assert(counter.rounds() == 1);
    assert(counter.rounds() == reference.rounds());
  }

  // 4. A fan-out send to two peers, then the reply, is two rounds and not
  //    three: the two sends belong to the phase that precedes the reply.
  reset_armed(counter);
  counter.received();          // phase 1: the primaries' a_tilde reaches P2
  counter.sent(); counter.sent();
  counter.received();          // phase 2: the reply
  assert(counter.rounds() == 2);

  // 5. An announced exchange is exactly one round however its two threads
  //    interleave, which is the property the socket-level rule cannot give.
  //    Run it both ways round and then concurrently, many times.
  reset_armed(counter);
  counter.begin_exchange();
  counter.sent();
  counter.received();
  counter.end_exchange();
  assert(counter.rounds() == 1);

  reset_armed(counter);
  counter.begin_exchange();
  counter.received();
  counter.sent();
  counter.end_exchange();
  assert(counter.rounds() == 1);

  for (std::size_t trial = 0; trial < 200; ++trial) {
    reset_armed(counter);
    counter.begin_exchange();
    std::thread sender([&counter] { counter.sent(); });
    std::thread receiver([&counter] { counter.received(); });
    sender.join();
    receiver.join();
    counter.end_exchange();
    assert(counter.rounds() == 1);
  }

  // 6. Announced exchanges compose with the socket-level rule: an exchange
  //    leaves the state "having received", so a following receive that is
  //    independent of it does not open a second round, and a following send
  //    does re-arm one.
  reset_armed(counter);
  counter.begin_exchange();
  counter.sent();
  counter.received();
  counter.end_exchange();
  counter.received();
  assert(counter.rounds() == 1);
  counter.sent();
  counter.received();
  assert(counter.rounds() == 2);

  // 7. One dot product per layer, as funcDotProductMPC runs it under
  //    PARALLEL_COMM: L layers are L rounds, whatever the scheduler does.
  for (std::size_t layers = 1; layers <= 16; ++layers) {
    reset_armed(counter);
    CausalRoundCounter reference;
    for (std::size_t i = 0; i < layers; ++i) {
      counter.begin_exchange();
      std::thread sender([&counter] { counter.sent(); });
      counter.received();
      sender.join();
      counter.end_exchange();
      const std::string previous = i ? "exchange" + std::to_string(i - 1) : "";
      reference.event("exchange" + std::to_string(i), true,
                      previous.empty() ? std::vector<std::string>{}
                                       : std::vector<std::string>{previous});
    }
    assert(counter.rounds() == layers);
    assert(counter.rounds() == reference.rounds());
  }

  // 8. Nothing is counted while the counter is disarmed, which is how the
  //    setup barrier and the tunnels stay out of the online figure.
  counter.reset();
  counter.sent();
  counter.received();
  counter.begin_exchange();
  counter.received();
  counter.end_exchange();
  assert(counter.rounds() == 0);

  // 9. Arming mid-flight starts a fresh count rather than inheriting state.
  counter.arm(true);
  counter.received();
  assert(counter.rounds() == 1);

  return 0;
}
