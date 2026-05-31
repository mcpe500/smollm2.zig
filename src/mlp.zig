//! SwiGLU Feed-Forward Network

const std = @import("std");
const tensor = @import("tensor.zig");

pub fn forward(
    gate_out: []f32,
    up_out: []f32,
    input: []const f32,
    gate_weight: []const f32,
    up_weight: []const f32,
    down_weight: []const f32,
    down_out: []f32,
    dim: u32,
    hidden_dim: u32,
) void {
    // gate = input @ gate_proj.T
    var i: u32 = 0;
    while (i < hidden_dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < dim) : (j += 1) {
            sum += input[j] * gate_weight[j * hidden_dim + i];
        }
        gate_out[i] = sum;
    }

    // up = input @ up_proj.T
    i = 0;
    while (i < hidden_dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < dim) : (j += 1) {
            sum += input[j] * up_weight[j * hidden_dim + i];
        }
        up_out[i] = sum;
    }

    // SiLU activation on gate
    i = 0;
    while (i < hidden_dim) : (i += 1) {
        gate_out[i] = tensor.silu(gate_out[i]);
    }

    // Multiply: gate *= up (element-wise)
    i = 0;
    while (i < hidden_dim) : (i += 1) {
        gate_out[i] *= up_out[i];
    }

    // down = gate @ down_proj.T
    i = 0;
    while (i < dim) : (i += 1) {
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < hidden_dim) : (j += 1) {
            sum += gate_out[j] * down_weight[i * hidden_dim + j];
        }
        down_out[i] = sum;
    }
}
