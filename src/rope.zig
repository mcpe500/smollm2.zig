//! Rotary Position Embedding (RoPE) implementation

const std = @import("std");

pub fn applyRoPE(
    q: []f32,
    k: []f32,
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_theta: f32,
) void {
    const half = head_dim / 2;
    var freqs = [_]f32{0} ** 32;

    // Precompute frequency for this position
    var i: u32 = 0;
    while (i < half) : (i += 1) {
        const freq = rope_theta * std.math.exp(-@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
        freqs[i] = @as(f32, @floatFromInt(pos)) * freq;
    }

    // Apply to query heads
    var h: u32 = 0;
    while (h < n_heads) : (h += 1) {
        const base = h * head_dim;
        var j: u32 = 0;
        while (j < half) : (j += 1) {
            const cos_val = @cos(freqs[j]);
            const sin_val = @sin(freqs[j]);
            const x0 = q[base + j];
            const x1 = q[base + j + half];
            q[base + j] = x0 * cos_val - x1 * sin_val;
            q[base + j + half] = x0 * sin_val + x1 * cos_val;
        }
    }

    // Apply to key heads
    h = 0;
    while (h < n_kv_heads) : (h += 1) {
        const base = h * head_dim;
        var j: u32 = 0;
        while (j < half) : (j += 1) {
            const cos_val = @cos(freqs[j]);
            const sin_val = @sin(freqs[j]);
            const x0 = k[base + j];
            const x1 = k[base + j + half];
            k[base + j] = x0 * cos_val - x1 * sin_val;
            k[base + j + half] = x0 * sin_val + x1 * cos_val;
        }
    }
}
