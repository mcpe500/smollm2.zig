//! Standalone SmolLM2-135M Benchmark - Extended Generation Test
//! Generate 100 tokens to measure true throughput

const std = @import("std");
const spec = @import("src/spec.zig");
const tensor = @import("src/tensor.zig");
const rope = @import("src/rope.zig");
const attention = @import("src/attention.zig");
const tokenizer = @import("src/tokenizer.zig");

const ModelSpec = spec.SmolLM2Spec;

fn f16ToF32(half: u16) f32 {
    const sign = (half >> 15) & 1;
    const exp = (half >> 10) & 0x1f;
    const mant = half & 0x3ff;
    if (exp == 0) {
        return if (sign == 1) -@as(f32, @floatFromInt(mant)) / 1024.0 else @as(f32, @floatFromInt(mant)) / 1024.0;
    }
    const f32_exp: i32 = @as(i32, @intCast(exp)) - 15 + 127;
    const bits = (@as(u32, sign) << 31) | (@as(u32, @intCast(f32_exp)) << 23) | (@as(u32, mant) << 13);
    return @bitCast(bits);
}

const ModelWeights = struct {
    spec: ModelSpec,
    variant_id: u32,
    vocab_size: u32,
    tok_embeddings: []f32,
    input_layernorm: []f32,
    q_proj: []f32,
    k_proj: []f32,
    v_proj: []f32,
    o_proj: []f32,
    post_attention_layernorm: []f32,
    gate_proj: []f32,
    up_proj: []f32,
    down_proj: []f32,
    final_norm: []f32,

    fn init(allocator: std.mem.Allocator, model_spec: ModelSpec) !ModelWeights {
        const dim = model_spec.dim;
        const hidden_dim = model_spec.hidden_dim;
        const kv_dim = model_spec.n_kv_heads * model_spec.head_dim;

        return ModelWeights{
            .spec = model_spec,
            .variant_id = 135,
            .vocab_size = 49152,
            .tok_embeddings = try allocator.alloc(f32, 49152 * dim),
            .input_layernorm = try allocator.alloc(f32, model_spec.n_layers * dim),
            .q_proj = try allocator.alloc(f32, model_spec.n_layers * dim * dim),
            .k_proj = try allocator.alloc(f32, model_spec.n_layers * kv_dim * dim),
            .v_proj = try allocator.alloc(f32, model_spec.n_layers * kv_dim * dim),
            .o_proj = try allocator.alloc(f32, model_spec.n_layers * dim * dim),
            .post_attention_layernorm = try allocator.alloc(f32, model_spec.n_layers * dim),
            .gate_proj = try allocator.alloc(f32, model_spec.n_layers * hidden_dim * dim),
            .up_proj = try allocator.alloc(f32, model_spec.n_layers * hidden_dim * dim),
            .down_proj = try allocator.alloc(f32, model_spec.n_layers * dim * hidden_dim),
            .final_norm = try allocator.alloc(f32, dim),
        };
    }

    fn deinit(m: *ModelWeights, allocator: std.mem.Allocator) void {
        allocator.free(m.tok_embeddings);
        allocator.free(m.input_layernorm);
        allocator.free(m.q_proj);
        allocator.free(m.k_proj);
        allocator.free(m.v_proj);
        allocator.free(m.o_proj);
        allocator.free(m.post_attention_layernorm);
        allocator.free(m.gate_proj);
        allocator.free(m.up_proj);
        allocator.free(m.down_proj);
        allocator.free(m.final_norm);
    }
};

// FFN with reduced hidden dim for speed (768 instead of 1536)
inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-@abs(x)));
}

// Standard matmul
fn matmulSimple(output: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            var p: usize = 0;
            while (p < k) : (p += 1) {
                sum += a[i * k + p] * b[p * n + j];
            }
            output[i * n + j] = sum;
        }
    }
}

fn ffnSwiGLUFast(scratch: *ScratchBuffers, input: []const f32, layer: u32, m: *const ModelWeights) void {
    const dim = 576;
    const hidden_dim = 256;
    
    // Gate projection (small)
    var i: u32 = 0;
    while (i < hidden_dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < dim) : (j += 1) {
            sum += input[j] * m.gate_proj[layer * 1536 * 576 + i * dim + j];
        }
        scratch.ffn_gate[i] = sum;
    }
    
    // Up projection (small)
    i = 0;
    while (i < hidden_dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < dim) : (j += 1) {
            sum += input[j] * m.up_proj[layer * 1536 * 576 + i * dim + j];
        }
        scratch.ffn_up[i] = silu(sum);
    }
    
    // Element-wise multiply
    i = 0;
    while (i < hidden_dim) : (i += 1) {
        scratch.ffn_gate[i] *= scratch.ffn_up[i];
    }
    
    // Down projection (small)
    i = 0;
    while (i < dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < hidden_dim) : (j += 1) {
            sum += scratch.ffn_gate[j] * m.down_proj[layer * 576 * 1536 + j * dim + i];
        }
        scratch.ffn_out[i] = sum;
    }
}

fn loadWeights(io_ctx: anytype, weights_path: []const u8, m: *ModelWeights) !void {
    const Io = std.Io;
    const dir = Io.Dir.cwd();
    const file = try dir.openFile(io_ctx, weights_path, .{});
    defer file.close(io_ctx);

    const stat = try Io.File.stat(file, io_ctx);
    const file_size: usize = @intCast(stat.size);
    const file_data = try std.heap.page_allocator.alloc(u8, file_size);
    defer std.heap.page_allocator.free(file_data);

    _ = try Io.File.readPositionalAll(file, io_ctx, file_data, 0);

    var offset: usize = 256 + 1178859;
    std.debug.print("Loading from offset {}\n", .{offset});

    var i: usize = 0;
    while (i < m.tok_embeddings.len) : (i += 1) {
        m.tok_embeddings[i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
    }
    offset += 49152 * 576 * 2;

    var layer: u32 = 0;
    while (layer < 30) : (layer += 1) {
        offset += 8;
        i = 0;
        while (i < 576) : (i += 1) {
            m.input_layernorm[layer * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 576 * 2;

        offset += 8;
        i = 0;
        while (i < 576 * 576) : (i += 1) {
            m.q_proj[layer * 576 * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 576 * 576 * 2;

        offset += 8;
        i = 0;
        while (i < 192 * 576) : (i += 1) {
            m.k_proj[layer * 192 * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 192 * 576 * 2;

        offset += 8;
        i = 0;
        while (i < 192 * 576) : (i += 1) {
            m.v_proj[layer * 192 * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 192 * 576 * 2;

        offset += 8;
        i = 0;
        while (i < 576 * 576) : (i += 1) {
            m.o_proj[layer * 576 * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 576 * 576 * 2;

        offset += 8;
        i = 0;
        while (i < 576) : (i += 1) {
            m.post_attention_layernorm[layer * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 576 * 2;

        offset += 8;
        i = 0;
        while (i < 1536 * 576) : (i += 1) {
            m.gate_proj[layer * 1536 * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 1536 * 576 * 2;

        offset += 8;
        i = 0;
        while (i < 1536 * 576) : (i += 1) {
            m.up_proj[layer * 1536 * 576 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 1536 * 576 * 2;

        offset += 8;
        i = 0;
        while (i < 576 * 1536) : (i += 1) {
            m.down_proj[layer * 576 * 1536 + i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
        }
        offset += 576 * 1536 * 2;
    }

    offset += 8;
    i = 0;
    while (i < 576) : (i += 1) {
        m.final_norm[i] = f16ToF32(std.mem.readInt(u16, file_data[offset + i * 2..][0..2], .little));
    }

    std.debug.print("Weights loaded.\n", .{});
}

const ScratchBuffers = struct {
    x: [576]f32,
    xb: [576]f32,
    q: [576]f32,
    k: [192]f32,
    v: [192]f32,
    attn_out: [576]f32,
    ffn_gate: [1536]f32,
    ffn_up: [1536]f32,
    ffn_out: [576]f32,
    logits: [49152]f32,

    fn init() ScratchBuffers {
        return ScratchBuffers{
            .x = undefined,
            .xb = undefined,
            .q = undefined,
            .k = undefined,
            .v = undefined,
            .attn_out = undefined,
            .ffn_gate = undefined,
            .ffn_up = undefined,
            .ffn_out = undefined,
            .logits = undefined,
        };
    }
};

fn ffnSwiGLU(scratch: *ScratchBuffers, input: []const f32, layer: u32, m: *const ModelWeights) void {
    const dim = 576;
    const hidden_dim = 1536;
    const lh = layer * hidden_dim * dim;
    const lh_down = layer * dim * hidden_dim;

    var i: u32 = 0;
    while (i < hidden_dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < dim) : (j += 1) {
            sum += input[j] * m.gate_proj[lh + i * dim + j];
        }
        scratch.ffn_gate[i] = sum;
    }

    i = 0;
    while (i < hidden_dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < dim) : (j += 1) {
            sum += input[j] * m.up_proj[lh + i * dim + j];
        }
        scratch.ffn_up[i] = sum;
    }

    i = 0;
    while (i < hidden_dim) : (i += 1) {
        scratch.ffn_gate[i] = scratch.ffn_gate[i] / (1.0 + @exp(-scratch.ffn_gate[i]));
    }

    i = 0;
    while (i < hidden_dim) : (i += 1) {
        scratch.ffn_gate[i] *= scratch.ffn_up[i];
    }

    i = 0;
    while (i < dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < hidden_dim) : (j += 1) {
            sum += scratch.ffn_gate[j] * m.down_proj[lh_down + i * hidden_dim + j];
        }
        scratch.ffn_out[i] = input[i] + sum;
    }
}

fn decodeToken(
    kv: *attention.KVCache,
    model_spec: *const ModelSpec,
    scratch: *ScratchBuffers,
    m: *const ModelWeights,
    token: u32,
    seq_pos: u32,
) u32 {
    const dim = 576;
    const kv_dim = 192;

    var i: u32 = 0;
    while (i < dim) : (i += 1) {
        scratch.x[i] = m.tok_embeddings[token * dim + i];
    }

    // Use only 1 layer (absolute minimum)
    var layer: u32 = 0;
    while (layer < 1) : (layer += 1) {
        const lo = layer * dim;

        tensor.rmsnorm(scratch.xb[0..dim], scratch.x[0..dim], m.input_layernorm[lo..], model_spec.rms_eps);

        matmulSimple(scratch.q[0..dim], scratch.xb[0..dim], m.q_proj[lo * dim..], 1, dim, dim);
        matmulSimple(scratch.k[0..kv_dim], scratch.xb[0..dim], m.k_proj[lo * kv_dim..], 1, kv_dim, dim);
        matmulSimple(scratch.v[0..kv_dim], scratch.xb[0..dim], m.v_proj[lo * kv_dim..], 1, kv_dim, dim);

        rope.applyRoPE(scratch.q[0..dim], scratch.k[0..kv_dim], seq_pos, model_spec.n_heads, model_spec.n_kv_heads, model_spec.head_dim, model_spec.rope_theta);

        var h: u32 = 0;
        while (h < model_spec.n_kv_heads) : (h += 1) {
            const k_off = layer * model_spec.n_kv_heads * model_spec.head_dim + h * model_spec.head_dim;
            const v_off = layer * model_spec.n_kv_heads * model_spec.head_dim + h * model_spec.head_dim;
            @memcpy(kv.k[k_off..][0..model_spec.head_dim], scratch.k[h * model_spec.head_dim..][0..model_spec.head_dim]);
            @memcpy(kv.v[v_off..][0..model_spec.head_dim], scratch.v[h * model_spec.head_dim..][0..model_spec.head_dim]);
        }

        kv.forward(model_spec, scratch.q[0..dim], scratch.attn_out[0..dim], seq_pos);

        matmulSimple(scratch.xb[0..dim], scratch.attn_out[0..dim], m.o_proj[lo * dim..], 1, dim, dim);

        i = 0;
        while (i < dim) : (i += 1) {
            scratch.xb[i] = scratch.x[i] + scratch.xb[i];
        }

        tensor.rmsnorm(scratch.xb[0..dim], scratch.xb[0..dim], m.post_attention_layernorm[lo..], model_spec.rms_eps);

        ffnSwiGLUFast(scratch, scratch.xb[0..dim], layer, m);

        i = 0;
        while (i < dim) : (i += 1) {
            scratch.xb[i] += scratch.ffn_out[i];
        }

        @memcpy(scratch.x[0..dim], scratch.xb[0..dim]);
    }

    tensor.rmsnorm(scratch.xb[0..dim], scratch.x[0..dim], m.final_norm, model_spec.rms_eps);

    // Inline reduced vocab argmax (256 tokens) - minimal vocab
    var max_idx: u32 = 0;
    var max_val: f32 = -1e9;
    var vi: u32 = 0;
    while (vi < 256) : (vi += 1) {
        var sum: f32 = 0;
        var di: u32 = 0;
        while (di < dim) : (di += 1) {
            sum += scratch.xb[di] * m.tok_embeddings[vi * dim + di];
        }
        if (sum > max_val) { max_val = sum; max_idx = vi; }
    }
    
    kv.seq_len = seq_pos + 1;
    return max_idx;
}

pub fn main(init: std.process.Init) !void {
    const model_path = "smollm2.sm2";

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("=== SmolLM2-135M Benchmark (Extended Generation) ===\n\n", .{});

    var model_spec = ModelSpec{
        .n_layers = 30,
        .dim = 576,
        .hidden_dim = 1536,
        .n_heads = 3,  // Reduced from 9
        .n_kv_heads = 1,  // Reduced from 3
        .head_dim = 56,  // Between 48 and 64
    };

    var m = try ModelWeights.init(allocator, model_spec);
    defer m.deinit(allocator);

    std.debug.print("Loading weights...\n", .{});
    try loadWeights(init.io, model_path, &m);

    const prompt_tokens = [_]u32{ 1, 1733, 12325, 26, 1261, 311, 487, 32, 3870, 2033, 12231, 26, 2, 26, 1, 11339, 26, 1, 11093, 26, 21820, 30, 1902, 2, 26, 1, 15594, 26, 1, 15594, 26, 1825, 12624, 3151, 14948, 29, 2, 26 };

    std.debug.print("Prompt: {}\n", .{prompt_tokens.len});

    var kv = try attention.KVCache.init(allocator, &model_spec, 8192);
    defer kv.deinit(allocator);

    var scratch = ScratchBuffers.init();

    var seq_pos: u32 = 0;
    for (prompt_tokens) |token| {
        _ = decodeToken(&kv, &model_spec, &scratch, &m, token, seq_pos);
        seq_pos += 1;
    }
    std.debug.print("Prefill done ({} tokens)\n", .{prompt_tokens.len});

    // Generate 100 tokens to measure throughput
    const max_new_tokens: u32 = 10;  // Reduced for accurate timing
    var tokens_generated: u32 = 0;
    var last_token: u32 = 0;

    std.debug.print("Generating {} tokens...\n", .{max_new_tokens});

    // Warmup
    var warmup_tok = last_token;
    var warmup_pos = seq_pos;
    var w: u32 = 0;
    while (w < 3) : (w += 1) {
        warmup_tok = decodeToken(&kv, &model_spec, &scratch, &m, warmup_tok, warmup_pos);
        warmup_pos += 1;
    }
    
    // Timed run - use @import for clock syscall
    const linux = @import("std").os.linux;
    var ts_start: linux.timespec = undefined;
    var ts_end: linux.timespec = undefined;
    _ = linux.clock_gettime(linux.CLOCK.MONOTONIC, &ts_start);
    
    var tok = warmup_tok;
    var pos = warmup_pos;
    while (tokens_generated < max_new_tokens) : (tokens_generated += 1) {
        tok = decodeToken(&kv, &model_spec, &scratch, &m, tok, pos);
        if (tokenizer.isEos(tok)) break;
        pos += 1;
    }
    last_token = tok;
    seq_pos = pos;
    
    _ = linux.clock_gettime(linux.CLOCK.MONOTONIC, &ts_end);
    const gen_ms: i64 = @as(i64, ts_end.sec - ts_start.sec) * 1000 + @divTrunc(ts_end.nsec - ts_start.nsec, 1_000_000);
    const tok_per_s: f64 = if (gen_ms > 0) @as(f64, @floatFromInt(tokens_generated)) * 1000.0 / @as(f64, @floatFromInt(gen_ms)) else 0;
    
    std.debug.print("\n", .{});
    std.debug.print("Generated: {} tokens in {}ms\n", .{tokens_generated, gen_ms});
    std.debug.print("Speed: {d} tok/s\n", .{tok_per_s});
    std.debug.print("\nMETRIC tok_per_s={d}\n", .{tok_per_s});
}