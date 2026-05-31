//! Forward pass inference

const std = @import("std");
const tensor = @import("tensor.zig");
const rope = @import("rope.zig");
const mlp = @import("mlp.zig");
const model = @import("model.zig");
const KVCache = @import("attention.zig").KVCache;

pub const InferenceContext = struct {
    model: *model.ModelWeights,
    kv_cache: KVCache,
    scratch: ScratchBuffers,

    pub fn init(m: *model.ModelWeights, allocator: std.mem.Allocator) !InferenceContext {
        const spec = &m.spec;
        const kv_cache = try KVCache.init(allocator, spec, spec.max_seq_len);
        errdefer kv_cache.deinit(allocator);

        const scratch = ScratchBuffers.init(allocator, spec.dim, spec.hidden_dim);
        errdefer scratch.deinit(allocator);

        return InferenceContext{
            .model = m,
            .kv_cache = kv_cache,
            .scratch = scratch,
        };
    }

    pub fn deinit(ctx: *InferenceContext, allocator: std.mem.Allocator) void {
        ctx.kv_cache.deinit(allocator);
        ctx.scratch.deinit(allocator);
    }

    pub fn prefill(ctx: *InferenceContext, tokens: []const u32) !void {
        for (tokens, 0..) |token, pos| {
            try ctx.decodeToken(token, @intCast(pos));
        }
    }

    pub fn decodeToken(ctx: *InferenceContext, token: u32, seq_pos: u32) !void {
        const model_weights = ctx.model;
        const spec = &model_weights.spec;
        const dim = spec.dim;
        const hidden_dim = spec.hidden_dim;
        const kv_dim = spec.n_kv_heads * spec.head_dim;

        // 1. Embedding lookup
        var i: u32 = 0;
        while (i < dim) : (i += 1) {
            ctx.scratch.x[i] = model_weights.tok_embeddings[token * dim + i];
        }

        // 2. Process each layer
        var layer: u32 = 0;
        while (layer < spec.n_layers) : (layer += 1) {
            const layer_offset = layer * dim;
            const layer_hid_offset = layer * hidden_dim;

            // Pre-norm
            tensor.rmsnorm(ctx.scratch.xb[0..dim], ctx.scratch.x[0..dim], model_weights.input_layernorm[layer_offset..], spec.rms_eps);

            // Q, K, V projections
            tensor.matmulTransposeB(ctx.scratch.q[0..dim], ctx.scratch.xb[0..dim], model_weights.q_proj[layer_offset * dim ..], dim, dim, dim);
            tensor.matmulTransposeB(ctx.scratch.k[0..kv_dim], ctx.scratch.xb[0..dim], model_weights.k_proj[layer_offset * kv_dim ..], kv_dim, dim, dim);
            tensor.matmulTransposeB(ctx.scratch.v[0..kv_dim], ctx.scratch.xb[0..dim], model_weights.v_proj[layer_offset * kv_dim ..], kv_dim, dim, dim);

            // Apply RoPE
            rope.applyRoPE(ctx.scratch.q[0..dim], ctx.scratch.k[0..kv_dim], seq_pos, spec.n_heads, spec.n_kv_heads, spec.head_dim, spec.rope_theta);

            // Store to KV cache
            var h: u32 = 0;
            while (h < spec.n_kv_heads) : (h += 1) {
                ctx.kv_cache.store(&ctx.model.spec, layer, h, seq_pos, ctx.scratch.k[h * spec.head_dim ..], ctx.scratch.v[h * spec.head_dim ..]);
            }

            // Attention
            ctx.kv_cache.forward(&ctx.model.spec, ctx.scratch.q[0..dim], ctx.scratch.attn_out[0..dim], seq_pos);

            // O projection
            tensor.matmulTransposeB(ctx.scratch.xb[0..dim], ctx.scratch.attn_out[0..dim], model_weights.o_proj[layer_offset * dim ..], dim, dim, dim);

            // Residual
            i = 0;
            while (i < dim) : (i += 1) {
                ctx.scratch.xb[i] = ctx.scratch.x[i] + ctx.scratch.xb[i];
            }

            // Post-attention norm
            tensor.rmsnorm(ctx.scratch.xb[0..dim], ctx.scratch.xb[0..dim], model_weights.post_attention_layernorm[layer_offset..], spec.rms_eps);

            // SwiGLU FFN
            mlp.forward(
                ctx.scratch.ffn_gate[0..hidden_dim],
                ctx.scratch.ffn_up[0..hidden_dim],
                ctx.scratch.xb[0..dim],
                model_weights.gate_proj[layer_hid_offset * dim ..],
                model_weights.up_proj[layer_hid_offset * dim ..],
                model_weights.down_proj[layer_hid_offset * hidden_dim ..],
                ctx.scratch.ffn_out[0..dim],
                dim,
                hidden_dim,
            );

            // Final residual
            i = 0;
            while (i < dim) : (i += 1) {
                ctx.scratch.xb[i] += ctx.scratch.ffn_out[i];
            }

            // Copy to x for next layer
            @memcpy(ctx.scratch.x[0..dim], ctx.scratch.xb[0..dim]);
        }

        // Final norm
        tensor.rmsnorm(ctx.scratch.xb[0..dim], ctx.scratch.x[0..dim], model_weights.final_norm, spec.rms_eps);

        // LM head
        tensor.matmulTransposeB(ctx.scratch.logits[0..model_weights.spec.vocab_size], ctx.scratch.xb[0..dim], model_weights.tok_embeddings, model_weights.spec.vocab_size, dim, dim);

        ctx.kv_cache.seq_len = seq_pos + 1;
    }
};

const ScratchBuffers = struct {
    x: []f32,
    xb: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    ffn_gate: []f32,
    ffn_up: []f32,
    ffn_out: []f32,
    logits: []f32,

    pub fn init(allocator: std.mem.Allocator, dim: u32, hidden_dim: u32, vocab_size: u32) !ScratchBuffers {
        return ScratchBuffers{
            .x = try allocator.alloc(f32, dim),
            .xb = try allocator.alloc(f32, dim),
            .q = try allocator.alloc(f32, dim),
            .k = try allocator.alloc(f32, 32 * 64), // max kv_dim
            .v = try allocator.alloc(f32, 32 * 64),
            .attn_out = try allocator.alloc(f32, dim),
            .ffn_gate = try allocator.alloc(f32, hidden_dim),
            .ffn_up = try allocator.alloc(f32, hidden_dim),
            .ffn_out = try allocator.alloc(f32, dim),
            .logits = try allocator.alloc(f32, vocab_size),
        };
    }

    pub fn deinit(sb: *ScratchBuffers, allocator: std.mem.Allocator) void {
        allocator.free(sb.x);
        allocator.free(sb.xb);
        allocator.free(sb.q);
        allocator.free(sb.k);
        allocator.free(sb.v);
        allocator.free(sb.attn_out);
        allocator.free(sb.ffn_gate);
        allocator.free(sb.ffn_up);
        allocator.free(sb.ffn_out);
        allocator.free(sb.logits);
    }
};
