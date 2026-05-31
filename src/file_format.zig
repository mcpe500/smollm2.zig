//! SM2 file format parser
//! Custom binary format for SmolLM2 models

const std = @import("std");
const SmolLM2Spec = @import("spec.zig").SmolLM2Spec;

pub const SM2Header = extern struct {
    magic: [8]u8,
    version: u32,
    variant_id: u32,
    quant_type: u32,
    flags: u32,

    vocab_size: u32,
    n_layers: u32,
    dim: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,

    rms_eps: f32,
    rope_theta: f32,

    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,

    tokenizer_offset: u64,
    tokenizer_size: u64,
    tensor_index_offset: u64,
    tensor_index_size: u64,
    weights_offset: u64,
    weights_size: u64,
    checksum: u64,
};

pub const QuantType = enum(u32) {
    f16 = 0,
    q8_0 = 1,
    q4_0 = 2,
    q4_k = 3,
};

pub const TensorMeta = struct {
    name: [64]u8,
    offset: u64,
    size: u64,
};

pub fn readHeader(file: std.fs.File) !SM2Header {
    var header: SM2Header = undefined;
    try file.readAll(std.mem.asBytes(&header));
    return header;
}

pub fn validateHeader(header: *const SM2Header) !void {
    if (!std.mem.eql(u8, &header.magic, "SM2C001")) {
        return error.InvalidMagic;
    }
    if (header.version != 1) {
        return error.UnsupportedVersion;
    }
}

pub fn loadHeader(path: []const u8) !SM2Header {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return readHeader(file);
}
