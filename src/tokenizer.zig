//! Byte-Pair Encoding Tokenizer for SmolLM2
//! Supports ChatML special tokens

const std = @import("std");

pub const SpecialTokens = struct {
    pub const BOS: u32 = 1;           // <|im_start|>
    pub const EOS: u32 = 2;           // <|im_end|>
    pub const PAD: u32 = 0;           // <|endoftext|>
    pub const NL: u32 = 198;          // Ċ (newline in vocab)
    pub const IM_END_ALT: u32 = 128009; // Alternate EOS token
};

pub fn isEos(token: u32) bool {
    return token == SpecialTokens.EOS or
        token == SpecialTokens.PAD or
        token == SpecialTokens.IM_END_ALT;
}

pub const Tokenizer = struct {
    vocab: [][]u8,
    token_to_id: std.AutoHashMap([]const u8, u32),
    vocab_size: u32,

    pub fn init(allocator: std.mem.Allocator, vocab_size: u32) !Tokenizer {
        const vocab = try allocator.alloc([]u8, vocab_size);
        errdefer allocator.free(vocab);
        for (vocab) |*v| v.* = &[_]u8{};

        var token_to_id = std.AutoHashMap([]const u8, u32).init(allocator);
        errdefer token_to_id.deinit();

        return Tokenizer{
            .vocab = vocab,
            .token_to_id = token_to_id,
            .vocab_size = vocab_size,
        };
    }

    pub fn deinit(t: *Tokenizer, allocator: std.mem.Allocator) void {
        for (t.vocab) |v| {
            allocator.free(v);
        }
        allocator.free(t.vocab);
        t.token_to_id.deinit();
    }

    pub fn decode(t: *Tokenizer, token_id: u32) ?[]const u8 {
        if (token_id >= t.vocab_size) return null;
        return t.vocab[token_id];
    }

    pub fn encode(t: *Tokenizer, text: []const u8, ids: *std.ArrayList(u32), allocator: std.mem.Allocator) !void {
        _ = allocator;
        var i: usize = 0;
        while (i < text.len) {
            // Simple byte-level encoding (placeholder for BPE)
            const byte = text[i];
            if (t.token_to_id.get(&[_]u8{byte})) |id| {
                try ids.append(id);
            } else {
                try ids.append(byte);
            }
            i += 1;
        }
    }
};

pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

pub fn formatChatPrompt(messages: []const ChatMessage, allocator: std.mem.Allocator) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    for (messages) |msg| {
        try buf.appendSlice("<|im_start|>");
        try buf.appendSlice(msg.role);
        try buf.append('\n');
        try buf.appendSlice(msg.content);
        try buf.append('\n');
        try buf.appendSlice("<|im_end|>");
        try buf.append('\n');
    }
    try buf.appendSlice("<|im_start|>assistant\n");

    return buf.toOwnedSlice();
}

pub fn extractAssistantResponse(full_text: []const u8) []const u8 {
    const marker = "<|im_start|>assistant\n";
    if (std.mem.indexOf(u8, full_text, marker)) |idx| {
        return full_text[idx + marker.len ..];
    }
    return full_text;
}
