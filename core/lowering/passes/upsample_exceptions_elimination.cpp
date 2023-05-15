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
struct UpsampleExceptionsPatternElimination {
  UpsampleExceptionsPatternElimination(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    findAndReplaceUpsampleNodes(graph_->block());
    torch::jit::EliminateDeadCode(graph_);
    LOG_GRAPH("Post upsample exceptions elimination: " << *graph_);
  }

 private:
  bool isUpsampleBlock(Block* b) {
    auto b_last = b->nodes().rbegin();
    return (*(b_last))->kind() == aten::upsample_bilinear2d;
  }

  bool isExceptionBlock(Block* b) {
    auto b_last = b->nodes().rbegin();
    return (*(b_last))->kind() == prim::RaiseException;
  }

  void copyAllNodes(Block* upsample_block, Block* root) {
    auto if_node = upsample_block->owningNode();

    constexpr bool copy_blocks = true;
    // Local map of return values of newly created node clones:
    // this allows to define correctly the inputs of subsequent nodes to be cloned
    std::unordered_map<Value*, Value*> local_map;
    auto env = [&local_map](Value* v) -> Value* {
      auto it = local_map.find(v);
      if (it != local_map.end()) {
        return it->second;
      }
      return v;
    };
    for (auto* upsample_block_node : upsample_block->nodes()) {
      // This part should do something similar to (but seems to work better):
      // upsample_block_node->output(no)->replaceAllUsesWith(new_node->output(no));
      auto* new_node = root->owningGraph()->createClone(upsample_block_node, env, copy_blocks)->insertBefore(if_node);
      for (size_t no = 0; no < upsample_block_node->outputs().size(); ++no) {
        auto old_output = upsample_block_node->outputs()[no];
        auto new_output = new_node->outputs()[no];
        local_map[old_output] = new_output;
        new_output->copyMetadata(old_output);
      }

      // If n outputs are in the upsample_block outputs, replace all uses of the corresponding If node output with
      // the new node outputs
      // Ordering of the block outputs and If node outputs should be the same
      for (size_t i = 0; i < upsample_block->outputs().size(); ++i) {
        auto pos = std::find(
            upsample_block_node->outputs().begin(), upsample_block_node->outputs().end(), upsample_block->outputs()[i]);
        if (pos != upsample_block_node->outputs().end()) {
          auto dist = std::distance(upsample_block_node->outputs().begin(), pos);
          if_node->output(i)->replaceAllUsesWith(new_node->output(dist));
        }
      }
    }
  }

  void findAndReplaceUpsampleNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto if_node = *it;
      if (if_node->kind() == prim::If && if_node->blocks().size() == 2) {
        Block* upsample_block = nullptr;
        if (isUpsampleBlock(if_node->blocks()[0]) && isExceptionBlock(if_node->blocks()[1])) {
          LOG_GRAPH("Found that node " << *if_node << "  is an Upsample node" << std::endl);
          upsample_block = if_node->blocks()[0];
        } else if (isUpsampleBlock(if_node->blocks()[1]) && isExceptionBlock(if_node->blocks()[0])) {
          LOG_GRAPH("Found that node " << *if_node << "  is an Upsample node" << std::endl);
          upsample_block = if_node->blocks()[1];
        }
        if (upsample_block) {
          copyAllNodes(upsample_block, b);
        }
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};
} // namespace

void EliminateUpsampleExceptionsPattern(std::shared_ptr<Graph> graph) {
  UpsampleExceptionsPatternElimination eppe(std::move(graph));
  eppe.run();
  if (graph) {
    LOG_GRAPH("Post Eliminate Upsample Exceptions: " << *graph);
  }
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
