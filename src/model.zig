//! Model loading and data structures

const std = @import("std");
const SmolLM2Spec = @import("spec.zig").SmolLM2Spec;
const tokenizer = @import("tokenizer.zig");

pub const ModelWeights = struct {
    spec: SmolLM2Spec,
    variant_id: u32 = 135,
    vocab_size: u32 = 49152,
    tokenizer: tokenizer.Tokenizer,

    pub fn init(allocator: std.mem.Allocator) !ModelWeights {
        var tok = try tokenizer.Tokenizer.init(allocator, 49152);
        errdefer tok.deinit(allocator);

        return ModelWeights{
            .spec = SmolLM2Spec{
                .n_layers = 30,
                .dim = 576,
                .hidden_dim = 1536,
                .n_heads = 9,
                .n_kv_heads = 3,
            },
            .tokenizer = tok,
        };
    }

    pub fn deinit(model: *ModelWeights, allocator: std.mem.Allocator) void {
        model.tokenizer.deinit(allocator);
    }
};
