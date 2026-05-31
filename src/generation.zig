//! Generation loop with proper EOS handling

const std = @import("std");
const inference = @import("inference.zig");
const tokenizer = @import("tokenizer.zig");

pub const GenerationConfig = struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.0,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repetition_penalty: f32 = 1.1,
};

pub const GenerationResult = struct {
    tokens: []u32,
    text: []u8,
    tok_per_sec: f32,
    stopped_early: bool,
};

pub fn generate(
    ctx: *inference.InferenceContext,
    prompt_tokens: []const u32,
    config: GenerationConfig,
    allocator: std.mem.Allocator,
) !GenerationResult {
    const start_time = std.time.milliTimestamp();

    // Prefill with prompt
    try ctx.prefill(prompt_tokens);

    var tokens = std.ArrayList(u32).init(allocator);
    errdefer tokens.deinit();
    try tokens.appendSlice(prompt_tokens);

    var seq_pos = @as(u32, @intCast(prompt_tokens.len));

    // Generate tokens
    while (tokens.items.len < prompt_tokens.len + config.max_tokens) : (seq_pos += 1) {
        // Sample next token from logits
        const token = sampleToken(ctx.scratch.logits, config, allocator);
        try tokens.append(token);

        // CRITICAL: Check ALL EOS tokens (Bug #1 fix)
        if (tokenizer.isEos(token)) {
            break;
        }

        // Decode next token
        try ctx.decodeToken(token, seq_pos);
    }

    const elapsed_ms = std.time.milliTimestamp() - start_time;
    const gen_tokens = tokens.items.len - prompt_tokens.len;
    const tok_per_sec = if (elapsed_ms > 0) @as(f32, @floatFromInt(gen_tokens)) / (@as(f32, @floatFromInt(elapsed_ms)) / 1000.0) else 0;

    // Decode tokens to text
    const text = try decodeTokens(ctx.model.tokenizer, tokens.items[prompt_tokens.len..], allocator);

    return GenerationResult{
        .tokens = try tokens.toOwnedSlice(),
        .text = text,
        .tok_per_sec = tok_per_sec,
        .stopped_early = tokenizer.isEos(tokens.items[tokens.items.len - 1]),
    };
}

fn sampleToken(logits: []f32, config: GenerationConfig, allocator: std.mem.Allocator) u32 {
    // Temperature scaling
    if (config.temperature > 0) {
        var max_logit: f32 = -1e9;
        for (logits) |l| {
            if (l > max_logit) max_logit = l;
        }
        for (logits) |*l| {
            l.* = (l.* - max_logit) / config.temperature;
        }
    }

    // Top-k filtering
    var filtered_logits = std.ArrayList(struct { idx: u32, val: f32 }).init(allocator);
    defer filtered_logits.deinit();

    if (config.top_k > 0) {
        var i: u32 = 0;
        while (i < filtered_logits.capacity) : (i += 1) {
            if (@as(u32, @intCast(i)) >= logits.len) break;
            filtered_logits.appendAssumeCapacity(.{ .idx = i, .val = logits[i] });
        }
        // Partial sort for top-k
        std.sort.sort(struct { idx: u32, val: f32 }, filtered_logits.items, {}, struct {
            fn less(_: void, a: struct { idx: u32, val: f32 }, b: struct { idx: u32, val: f32 }) bool {
                return a.val > b.val;
            }
        }.less);
        const cutoff = if (config.top_k < filtered_logits.items.len) filtered_logits.items[config.top_k].val else -1e9;
        for (logits) |*l| {
            if (l.* < cutoff) l.* = -1e9;
        }
    }

    // Sample from distribution
    var sum: f32 = 0;
    for (logits) |l| {
        const exp_l = @exp(l);
        sum += exp_l;
    }

    const r = @as(f32, @floatFromInt(std.rand.Random.defaultPrng.init(0).int(u32))) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
    var cumsum: f32 = 0;
    var i: u32 = 0;
    while (i < logits.len) : (i += 1) {
        cumsum += @exp(logits[i]) / sum;
        if (r <= cumsum) return i;
    }
    return logits.len - 1;
}

fn decodeTokens(tok: *tokenizer.Tokenizer, tokens: []const u32, allocator: std.mem.Allocator) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    for (tokens) |t| {
        if (tokenizer.isEos(t)) break;
        if (t == 1) continue; // <|im_start|>
        if (tok.decode(t)) |text| {
            try buf.appendSlice(text);
        }
    }

    return buf.toOwnedSlice();
}
