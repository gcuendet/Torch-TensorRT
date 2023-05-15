#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/runtime/graph_executor.h"

#include "core/util/prelude.h"

#include <vector>

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {
namespace {
using namespace torch::jit;
struct ExceptionOrPassPatternElimination {
  ExceptionOrPassPatternElimination(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    findExceptionOrPassNodes(graph_->block());
    torch::jit::EliminateDeadCode(graph_);
    LOG_GRAPH("Post exeception or pass elimination: " << *graph_);
  }

 private:
  bool isExceptionOrPassNode(Node* n) {
    if (!(n->kind() == prim::If && n->blocks().size() == 2)) {
      return false;
    }

    // Make sure that the node doesn't actually produce any Value that are
    // used by other nodes
    if (n->outputs().size() != 0)
      return false;

    const auto arm1 = n->blocks()[0];
    const auto arm2 = n->blocks()[1];

    auto arm1_last = arm1->nodes().rbegin();
    auto arm2_last = arm2->nodes().rbegin();

    const bool arm1_ends_with_exception = (*arm1_last)->kind() == prim::RaiseException;
    const bool next_arm1_ends_with_return = (*std::next(arm1_last))->kind() == prim::Return;

    const bool arm2_ends_with_exception = (*arm2_last)->kind() == prim::RaiseException;
    const bool next_arm2_ends_with_return = (*std::next(arm2_last))->kind() == prim::Return;

    if (!arm1_ends_with_exception && !arm2_ends_with_exception) {
      // Neither arm matches the pattern
      return false;
    }

    // Check that the arm that does not raise also does not contain any computation, but basically
    // just a prim::Return
    if (arm1_ends_with_exception && !next_arm2_ends_with_return) {
      return false;
    } else if (arm2_ends_with_exception && !next_arm1_ends_with_return) {
      return false;
    }

    return true;
  }

  void findExceptionOrPassNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (isExceptionOrPassNode(n)) {
        LOG_GRAPH("Found that node " << *n << "  is an exception or pass node (EliminateChecks)" << std::endl);
        it.destroyCurrent();
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};
} // namespace

void EliminateExceptionOrPassPattern(std::shared_ptr<Graph> graph) {
  ExceptionOrPassPatternElimination eppe(std::move(graph));
  eppe.run();
  if (graph) {
    LOG_GRAPH("Post Eliminate Exception or Pass Patterns: " << *graph);
  }
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
