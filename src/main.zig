//! SmolLM2-135M Inference Engine in Zig
//! Phase 1: Basic correctness with scalar operations

const std = @import("std");
const model = @import("model.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("SmolLM2-135M Zig Inference Engine\n", .{});
    std.debug.print("Phase 1: Basic implementation\n", .{});
    std.debug.print("==============================\n\n", .{});

    // Create model with placeholder weights
    std.debug.print("Initializing model...\n", .{});
    var model_weights = try model.ModelWeights.init(allocator);
    defer model_weights.deinit(allocator);
    std.debug.print("Model initialized with spec:\n", .{});
    std.debug.print("  - Layers: {}\n", .{model_weights.spec.n_layers});
    std.debug.print("  - Dim: {}\n", .{model_weights.spec.dim});
    std.debug.print("  - Heads: {} ({} KV)\n", .{model_weights.spec.n_heads, model_weights.spec.n_kv_heads});
    std.debug.print("  - Hidden: {}\n\n", .{model_weights.spec.hidden_dim});

    std.debug.print("Phase 1 Complete!\n", .{});
    std.debug.print("Next: Implement model loading from .sm2 files\n", .{});
}
