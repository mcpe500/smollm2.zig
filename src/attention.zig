//! Grouped-Query Attention (GQA) with KV cache

const std = @import("std");
const tensor = @import("tensor.zig");
const rope = @import("rope.zig");
const SmolLM2Spec = @import("spec.zig").SmolLM2Spec;

pub const KVCache = struct {
    k: []f32,
    v: []f32,
    seq_len: u32,

    pub fn init(allocator: std.mem.Allocator, spec: *const SmolLM2Spec, max_seq: u32) !KVCache {
        const kv_dim = spec.n_kv_heads * spec.head_dim;
        const size = spec.n_layers * max_seq * kv_dim;
        const k = try allocator.alloc(f32, size);
        errdefer allocator.free(k);
        const v = try allocator.alloc(f32, size);
        errdefer allocator.free(v);
        @memset(k, 0);
        @memset(v, 0);
        return KVCache{ .k = k, .v = v, .seq_len = 0 };
    }

    pub fn deinit(cache: *KVCache, allocator: std.mem.Allocator) void {
        allocator.free(cache.k);
        allocator.free(cache.v);
    }

    fn kvOffset(cache: *const KVCache, spec: *const SmolLM2Spec, layer: u32, pos: u32) usize {
        const kv_dim = spec.n_kv_heads * spec.head_dim;
        _ = cache;
        return layer * spec.n_layers * kv_dim + pos * spec.head_dim;
    }

    pub fn store(cache: *KVCache, spec: *const SmolLM2Spec, layer: u32, head: u32, pos: u32, k_in: []const f32, v_in: []const f32) void {
        const off = cache.kvOffset(spec, layer, pos);
        @memcpy(cache.k[off .. off + spec.head_dim], k_in);
        @memcpy(cache.v[off .. off + spec.head_dim], v_in);
        _ = head;
    }

    pub fn forward(
        cache: *const KVCache,
        spec: *const SmolLM2Spec,
        q: []const f32,
        output: []f32,
        seq_pos: u32,
    ) void {
        const n_heads = spec.n_heads;
        const n_kv_heads = spec.n_kv_heads;
        const head_dim = spec.head_dim;
        const group_size = n_heads / n_kv_heads;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const kv_dim = n_kv_heads * head_dim;

        var qh: u32 = 0;
        while (qh < n_heads) : (qh += 1) {
            const kv_head = qh / group_size;
            const q_off = qh * head_dim;
            const out_off = qh * head_dim;

            // Compute attention scores
            var max_score: f32 = -1e9;
            const max_pos = seq_pos + 1;
            var scores = [_]f32{0} ** 8192;
            var pos: u32 = 0;
            while (pos < max_pos) : (pos += 1) {
                const k_off = cache.kvOffset(cache, spec, 0, pos);
                var score: f32 = 0;
                var d: u32 = 0;
                while (d < head_dim) : (d += 1) {
                    score += q[q_off + d] * cache.k[k_off + d];
                }
                score *= scale;
                scores[pos] = score;
                if (score > max_score) max_score = score;
            }

            // Softmax
            var sum_exp: f32 = 0;
            pos = 0;
            while (pos < max_pos) : (pos += 1) {
                scores[pos] = @exp(scores[pos] - max_score);
                sum_exp += scores[pos];
            }

            // Weighted sum of values
            pos = 0;
            while (pos < max_pos) : (pos += 1) {
                const v_off = cache.kvOffset(cache, spec, 0, pos);
                const attn = scores[pos] / sum_exp;
                var d2: u32 = 0;
                while (d2 < head_dim) : (d2 += 1) {
                    output[out_off + d2] += attn * cache.v[v_off + d2];
                }
            }
            _ = kv_dim;
            _ = kv_head;
        }
    }
};
