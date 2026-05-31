//! Basic tensor operations (Phase 1: scalar implementation)
//! Phase 2+ will add SIMD via @Vector

const std = @import("std");

pub fn matmul(
    output: []f32,
    a: []const f32,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    // output[m,n] = a[m,k] * b[k,n] (row-major)
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

pub fn matmulTransposeB(
    output: []f32,
    a: []const f32,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    // output[m,n] = a[m,k] * b[n,k] (b is column-major / transposed)
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            var p: usize = 0;
            while (p < k) : (p += 1) {
                sum += a[i * k + p] * b[j * k + p];
            }
            output[i * n + j] = sum;
        }
    }
}

pub fn rmsnorm(output: []f32, input: []const f32, weight: []const f32, eps: f32) void {
    const size = input.len;
    var sum_sq: f32 = 0;
    for (input) |x| {
        sum_sq += x * x;
    }
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(size)) + eps);
    for (output, input, weight) |*out, inp, w| {
        out.* = (inp / rms) * w;
    }
}

pub fn softmax(x: []f32) void {
    var max_val: f32 = -1e9;
    for (x) |v| {
        if (v > max_val) max_val = v;
    }
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    for (x) |*v| {
        v.* /= sum;
    }
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
