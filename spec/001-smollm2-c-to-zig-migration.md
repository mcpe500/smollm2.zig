# SmolLM2 C to Zig Migration Specification

**Document ID:** 001  
**Status:** Draft  
**Target:** smollm2.c -> smollm2.zig  
**Date:** 2026-05-30

---

## 1. Root Cause Analysis: Why smollm2.c Fails

### 1.1 Critical Bugs Identified

#### Bug #1: EOS Token Mishandling (Infinite Generation Loop)
**Severity:** Critical  
**Location:** `smollm2.c` main inference loop, `sm2_sampling.c`

**Problem:** The C implementation fails to properly handle End-of-Sequence tokens, resulting in continuous generation loops up to the maximum context size (8191 characters) for simple inputs like `"hello"`.

**Root Cause:**
```c
// smollm2.c line 218
if (token < 3) break; // EOS check - INSUFFICIENT

// Problem: Token ID 128009 (<|im_end|>) is the actual EOS token
// The model uses ChatML template where:
//   - Token 1: <|im_start|>
//   - Token 2: <|im_end|>
//   - Token 128009: <|im_end|> (alternate representation)
```

**Fix Required:**
```zig
// In Zig, we must check for ALL EOS variants:
const eos_tokens = [_]u32{ 0, 1, 2, 128009 };
fn is_eos(token: u32) bool {
    return for (eos_tokens) |eos| {
        if (token == eos) break true;
    } else false;
}
```

#### Bug #2: Unvectorized Operations (7.1 tok/s Performance)
**Severity:** High  
**Location:** `sm2_context.c` matrix operations

**Problem:** Processing speed drops to unacceptable 7.1 tok/s due to:
- Poor loop alignment
- Missing SIMD vectorization
- Unoptimized attention computation

**Root Cause:**
```c
// sm2_context.c line 332 - Naive matrix multiply
for (int i = 0; i < dim; i++) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    // 8x unrolling is insufficient without SIMD
    ...
}
```

**Fix Required:**
```zig
// Zig @Vector SIMD implementation
const SimdFloat = @Vector(8, f32);
fn matmul_vectorized(out: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    var i: usize = 0;
    while (i + 7 < m * n) : (i += 8) {
        const result = a_simd * b_simd; // SIMD multiply
        out[i..i+8] = result;
    }
}
```

#### Bug #3: Memory Invalidation in KV Cache
**Severity:** High  
**Location:** `sm2_context.c` attention_with_cache()

**Problem:** C pointer casting and uncontrolled allocations cause buffer issues during autoregressive generation.

**Root Cause:**
```c
// sm2_context.c line 279 - Unsafe cache indexing
int cache_idx = kv_head * ctx->params.max_context * head_dim + pos * head_dim + d;
float vv = ctx->scratch.v_cache[layer][cache_idx];  // Potential OOB
```

**Fix Required:**
```zig
// Zig: Bound-checked array access with Comptime bounds
fn v_cache_at(ctx: *Context, layer: usize, head: usize, pos: usize, dim: usize) f32 {
    const idx = head * ctx.max_seq * head_dim + pos * head_dim + dim;
    if (idx >= ctx.v_cache.len) @panic("KV cache OOB");
    return ctx.v_cache[idx];
}
```

#### Bug #4: Sampling Configuration Issues
**Severity:** Medium  
**Location:** `sm2_sampling.c` sm2_sample_token()

**Problem:** Default sampling parameters produce garbage tokens instead of readable text.

**Root Cause:**
```c
// Before: temp=0.8, top_p=90, top_k=40 -> garbage
// After: temp=0, greedy -> readable output
// But greedy is suboptimal for creative generation
```

**Fix Required:**
```zig
// Proper temperature scaling with numerical stability
fn apply_temperature(logits: []f32, temp: f32) void {
    if (temp <= 0.0) return;  // Greedy - no change
    const max_logit = max(logits);
    for (i, logit) in logits |logit| {
        logits[i] = (logit - max_logit) / temp;  // Numerical stable
    }
}
```

---

## 2. Architecture Overview: Zig Implementation Design

### 2.1 Design Principles

1. **Type Safety:** Zig's comptime and optional types prevent buffer overflows
2. **SIMD-First:** Native `@Vector` types for all matrix operations
3. **No Hidden State:** All state explicit in structures, no global variables
4. **Resource Safety:** File handles, memory explicitly managed with defer

### 2.2 Module Structure

```
src/
├── main.zig              # CLI entry point, argument parsing
├── model.zig             # Model loading, weight structures
├── spec.zig              # Model specifications (135M, 360M, 1.7B)
├── tensor.zig            # Linear algebra primitives, SIMD kernels
├── tokenizer.zig         # BPE tokenizer, ChatML template
├── inference.zig          # Forward pass, layer implementation
├── generation.zig       # Generation loop, sampling
├── attention.zig         # GQA attention, KV cache
├── rope.zig              # Rotary Position Embedding
├── mlp.zig               # SwiGLU feed-forward network
├── quant.zig             # Q4_K, Q8_0 quantization support
├── file_format.zig       # .sm2, GGUF file parsing
└── interfaces/
    ├── cli.zig           # Interactive CLI
    ├── tui.zig           # Terminal UI (optional)
    └── api.zig           # HTTP API server (optional)
```

### 2.3 Core Data Structures

```zig
// Model specification - compile-time constants
pub const SmolLM2Spec = struct {
    n_layers: u32,
    dim: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32 = 64,
    vocab_size: u32 = 49152,
    max_seq_len: u32 = 8192,
    rope_theta: f32 = 100000.0,
    rms_eps: f32 = 1e-5,
};

pub const SmolLM2Variants = enum {
    sm2_135m,
    sm2_360m,
    sm2_1700m,
};

pub const spec_135m = SmolLM2Spec{
    .n_layers = 30,
    .dim = 576,
    .hidden_dim = 1536,
    .n_heads = 9,
    .n_kv_heads = 3,
};

pub const spec_360m = SmolLM2Spec{
    .n_layers = 32,
    .dim = 960,
    .hidden_dim = 2560,
    .n_heads = 15,
    .n_kv_heads = 5,
};

pub const spec_1700m = SmolLM2Spec{
    .n_layers = 24,
    .dim = 2048,
    .hidden_dim = 8192,
    .n_heads = 32,
    .n_kv_heads = 32,
};
```

---

## 3. Model Loading: GGUF/.sm2 Format Support

### 3.1 File Format Support

**Priority 1: .sm2 Native Format**
- 256-byte header
- Tokenizer section (1,178,859 bytes)
- Tensor index
- Weight data (F16/F32)

**Priority 2: GGUF Format**
- Standard format from Ollama/HuggingFace
- Self-describing with tensor metadata
- Memory-mapped friendly

### 3.2 Header Structure (.sm2)

```zig
pub const SM2Header = packed struct {
    magic: [8]u8,           // "SM2C001"
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
```

### 3.3 Weight Loading

```zig
pub const Model = struct {
    spec: SmolLM2Spec,
    variant: SmolLM2Variants,

    // Embeddings
    tok_embeddings: []f32,  // [vocab_size, dim] F32

    // Layer weights
    input_layernorm: []f32,           // [n_layers, dim]
    q_proj: []f32,                    // [n_layers, dim, dim]
    k_proj: []f32,                    // [n_layers, kv_dim, dim]
    v_proj: []f32,                    // [n_layers, kv_dim, dim]
    o_proj: []f32,                    // [n_layers, dim, dim]
    post_attention_layernorm: []f32,  // [n_layers, dim]
    gate_proj: []f32,                 // [n_layers, hidden_dim, dim]
    up_proj: []f32,                   // [n_layers, hidden_dim, dim]
    down_proj: []f32,                 // [n_layers, dim, hidden_dim]
    final_norm: []f32,                // [dim]

    // Tokenizer
    tokenizer: *Tokenizer,

    // KV Cache
    k_cache: []f32,  // [n_layers, n_kv_heads, max_seq, head_dim]
    v_cache: []f32,
};

pub fn load_model(path: []const u8) !*Model {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var header: SM2Header = undefined;
    _ = try file.readAll(std.mem.asBytes(&header));

    if (!std.mem.eql(u8, &header.magic, "SM2C001")) {
        return error.InvalidMagic;
    }

    // Load weights based on quant type
    const spec = get_spec(header.variant_id);
    var model = try alloc_model(spec);

    // Load tokenizer
    try file.seek(header.tokenizer_offset);
    model.tokenizer = try load_tokenizer(file, header.tokenizer_size);

    // Load weights
    try file.seek(header.weights_offset);
    try load_weights_f16(file, model);

    return model;
}
```

---

## 4. Tokenizer: BPE with Proper ChatML Template

### 4.1 Tokenizer Structure

```zig
pub const Tokenizer = struct {
    vocab_size: u32,
    tokens: [][]const u8,      // vocab_size token strings
    token_to_id: std.AutoHashMap([]const u8, u32),

    // BPE merges (sorted by frequency)
    merges: [][]const u8,

    // Byte-level mapping
    byte_to_token: [256]u32,   // byte -> vocab token ID
};

pub const SpecialTokens = struct {
    pub const BOS: u32 = 1;           // <|im_start|>
    pub const EOS: u32 = 2;           // <|im_end|>
    pub const PAD: u32 = 0;           // <|endoftext|>
    pub const IM_END_ALT: u32 = 128009; // Alternative EOS token
};

pub fn is_eos(token: u32) bool {
    return token == SpecialTokens.EOS or
           token == SpecialTokens.PAD or
           token == SpecialTokens.IM_END_ALT;
}
```

### 4.2 BPE Encoding

```zig
pub fn encode(tok: *Tokenizer, text: []const u8, ids: *std.ArrayList(u32)) !void {
    // 1. Pretokenize: split on whitespace
    var pieces = pre_tokenize(text);

    // 2. For each word, convert bytes and apply BPE merges
    for (pieces) |piece| {
        try bpe_encode_word(tok, piece.text, ids);
    }
}

fn bpe_encode_word(tok: *Tokenizer, word: []const u8, ids: *std.ArrayList(u32)) !void {
    if (tok.merges.len == 0) {
        // Fallback: byte-level
        for (word) |byte| {
            try ids.append(tok.byte_to_token[byte]);
        }
        return;
    }

    // Initialize segments (byte-level tokens)
    var segments = std.ArrayList([]const u8).init();
    defer segments.deinit();

    for (word) |byte| {
        try segments.append(&[_]u8{byte});
    }

    // Apply BPE merges iteratively
    while (segments.items.len > 1) {
        // Find best merge (lowest rank in merges list)
        var best_rank: usize = std.math.maxInt(usize);
        var best_idx: usize = 0;

        for (segments.items.len - 1) |i| {
            const merged = try std.fmt.allocPrint(segments.allocator,
                "{s}{s}", .{ segments.items[i], segments.items[i + 1] });

            const rank = find_merge_rank(tok.merges, merged);
            if (rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
        }

        if (best_rank == std.math.maxInt(usize)) break; // No valid merges

        // Apply merge
        const merged = try std.fmt.allocPrint(segments.allocator,
            "{s}{s}", .{ segments.items[best_idx], segments.items[best_idx + 1] });
        segments.items[best_idx] = merged;
        _ = segments.orderedRemove(best_idx + 1);
    }

    // Convert to token IDs
    for (segments.items) |seg| {
        if (tok.token_to_id.get(seg)) |id| {
            try ids.append(id);
        }
    }
}
```

### 4.3 ChatML Template

```zig
pub fn format_chat_prompt(messages: []const ChatMessage) ![]u8 {
    var buf = std.ArrayList(u8).init();

    for (messages) |msg| {
        try buf.appendSlice("<|im_start|>");
        try buf.appendSlice(msg.role);
        try buf.append('\n');

        // Handle special tokens like Ċ (newline) in content
        for (msg.content) |c| {
            if (c == '\n') {
                try buf.append(0xC4);
                try buf.append(0x8A);
            } else {
                try buf.append(c);
            }
        }

        try buf.appendSlice("<|im_end|>");
        try buf.append('\n');
    }

    try buf.appendSlice("<|im_start|>assistant\n");

    return buf.toOwnedSlice();
}

pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};
```

---

## 5. Inference Engine: Forward Pass Implementation

### 5.1 Layer Forward

```zig
pub fn layer_forward(
    ctx: *InferenceContext,
    layer: u32,
    input: []f32,  // [dim]
    output: []f32,  // [dim] - written to
    seq_pos: u32,
) void {
    const model = ctx.model;
    const spec = model.spec;

    // 1. RMSNorm on input
    var normalized = ctx.scratch;
    rmsnorm_inplace(normalized, input, model.input_layernorm[layer], spec.dim);

    // 2. Q/K/V projections
    var q = ctx.q;  // [dim]
    var k = ctx.k;  // [kv_dim]
    var v = ctx.v;  // [kv_dim]

    matmul(q, normalized, model.q_proj[layer], spec.dim, spec.dim);
    matmul(k, normalized, model.k_proj[layer], spec.kv_dim, spec.dim);
    matmul(v, normalized, model.v_proj[layer], spec.kv_dim, spec.dim);

    // 3. Apply RoPE
    apply_rope(q, k, seq_pos, spec.n_heads, spec.n_kv_heads, spec.head_dim, spec.rope_theta);

    // 4. Store K/V to cache
    const kv_offset = layer * spec.n_kv_heads * spec.max_seq_len * spec.head_dim;
    const pos_offset = seq_pos * spec.head_dim;

    for (kv_head in 0..spec.n_kv_heads) |h| {
        const cache_base = kv_offset + h * spec.max_seq_len * spec.head_dim;
        @memcpy(ctx.k_cache[cache_base + pos_offset..], k[h * spec.head_dim..]);
        @memcpy(ctx.v_cache[cache_base + pos_offset..], v[h * spec.head_dim..]);
    }

    // 5. Attention with KV cache
    attention(output, q, layer, ctx, spec);

    // 6. O projection
    var attn_out = ctx.attn_out;
    matmul(attn_out, output, model.o_proj[layer], spec.dim, spec.dim);

    // 7. Residual: output += input
    for (i in 0..spec.dim) |i| {
        output[i] = input[i] + attn_out[i];
    }

    // 8. Post-attention RMSNorm
    rmsnorm_inplace(normalized, output, model.post_attention_layernorm[layer], spec.dim);

    // 9. SwiGLU FFN
    ffn_swiglu(output, normalized, model, layer, ctx);

    // 10. Final residual: output += attn_result (before FFN)
    for (i in 0..spec.dim) |i| {
        output[i] += attn_out[i];
    }
}
```

### 5.2 RMSNorm

```zig
fn rmsnorm(output: []f32, input: []const f32, weight: []const f32, eps: f32) void {
    const size = input.len;

    // Compute sum of squares
    var sum_sq: f32 = 0;
    for (input) |x| {
        sum_sq += x * x;
    }

    // Compute RMS
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(size)) + eps);

    // Normalize and scale
    for (output, input, weight) |out, inp, w| {
        out = (inp / rms) * w;
    }
}
```

---

## 6. Generation Loop: Proper EOS Handling

### 6.1 Generation State

```zig
pub const GenerationState = struct {
    model: *Model,
    ctx: *InferenceContext,

    // Current position
    pos: u32,
    last_token: u32,

    // KV cache state
    kv_cache_len: u32,

    // Sampling parameters
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 0,
    repetition_penalty: f32 = 1.0,

    // RNG state
    rng_state: u64,

    // Generation limits
    max_tokens: u32 = 256,
};

pub const GenerationResult = struct {
    tokens: []u32,
    text: []u8,
    tok_per_sec: f32,
    stopped_early: bool,
};
```

### 6.2 Generation Loop

```zig
pub fn generate(
    state: *GenerationState,
    prompt_tokens: []const u32,
    output: *std.ArrayList(u32),
) !GenerationResult {
    const start_time = std.time.milliTimestamp();

    // Prefill: process prompt
    try prefill(state, prompt_tokens);

    // Autoregressive generation
    var tokens_generated: u32 = 0;

    while (tokens_generated < state.max_tokens) {
        // Decode next token
        const token = try decode_next(state);

        // CRITICAL: Check ALL EOS tokens
        if (is_eos(token)) {
            break;
        }

        try output.append(token);
        tokens_generated += 1;

        // Update last token for next iteration
        state.last_token = token;
    }

    const elapsed_ms = std.time.milliTimestamp() - start_time;
    const tok_per_sec = @as(f32, @floatFromInt(tokens_generated)) /
                        (@as(f32, @floatFromInt(elapsed_ms)) / 1000.0);

    return GenerationResult{
        .tokens = output.toOwnedSlice(),
        .tok_per_sec = tok_per_sec,
        .stopped_early = is_eos(tokens_generated),
    };
}
```

### 6.3 EOS Token Detection (Critical)

```zig
// All valid EOS tokens for SmolLM2
const EOS_TOKEN_IDS = [_]u32{
    0,      // <|endoftext|> - Pad/EOS
    2,      // <|im_end|> - Standard EOS
    128009, // <|im_end|> - Alternate (from vocab)
};

pub fn is_eos(token: u32) bool {
    inline for (EOS_TOKEN_IDS) |eos| {
        if (token == eos) return true;
    }
    return false;
}

// Also handle special stop sequences
fn should_stop(tokens: []const u32) bool {
    // Check for </s> or other stop markers
    if (tokens.len >= 2) {
        if (tokens[tokens.len - 1] == 2 and tokens[tokens.len - 2] == 198) {
            return true; // Ċ<|im_end|> = end of assistant message
        }
    }
    return false;
}
```

---

## 7. Attention: GQA with Proper KV Head Broadcasting

### 7.1 Grouped Query Attention (GQA)

```zig
pub fn attention(
    output: []f32,          // [dim] - query heads output
    q: []const f32,         // [dim] - all query heads
    layer: u32,
    ctx: *InferenceContext,
    spec: SmolLM2Spec,
) void {
    const n_heads = spec.n_heads;
    const n_kv_heads = spec.n_kv_heads;
    const head_dim = spec.head_dim;
    const group_size = n_heads / n_kv_heads;  // 3 for 135M

    const scale = 1.0 / @sqrt(@as(f32, head_dim));

    // For each query head, attend over KV cache
    for (qh in 0..n_heads) |qi| {
        // Map query head to KV head (broadcast)
        const kv_head = qi / group_size;

        const q_offset = qi * head_dim;
        const q_head = q[q_offset..q_offset + head_dim];

        // Compute attention scores for all cached positions
        var max_score: f32 = -1e9;
        var scores: [512]f32 = undefined;
        var n_pos = ctx.kv_cache_len + 1;

        for (pos in 0..n_pos) |p| {
            const k_base = ctx.k_cache_base(layer, kv_head, p);
            var score: f32 = 0;

            // Dot product (unrolled for SIMD)
            var d: usize = 0;
            while (d + 7 < head_dim) : (d += 8) {
                const qv = q_head[d..d+8];
                const kv = k_base[d..d+8];
                score += @reduce(.Add, qv * kv);
            }
            // Handle remainder
            for (d..head_dim) |i| {
                score += q_head[i] * k_base[i];
            }

            scores[p] = score * scale;
            if (score > max_score) max_score = score;
        }

        // Softmax
        var sum_exp: f32 = 0;
        for (scores[0..n_pos]) |*s| {
            s.* = @exp(s.* - max_score);
            sum_exp += s.*;
        }

        // Weighted sum of values
        const out_offset = qi * head_dim;
        @memset(output[out_offset..out_offset + head_dim], 0);

        for (pos in 0..n_pos) |p| {
            const v_base = ctx.v_cache_base(layer, kv_head, p);
            const attn_weight = scores[p] / sum_exp;

            for (d in 0..head_dim) |i| {
                output[out_offset + i] += attn_weight * v_base[i];
            }
        }
    }
}
```

---

## 8. RoPE: Correct Implementation

### 8.1 RoPE Theory

RoPE (Rotary Position Embedding) encodes position by rotating query and key vectors:
- Each dimension pair (i, i+dim/2) is rotated by angle `theta^(-2i/dim)`
- Allows attention to use relative positions without explicit encoding

### 8.2 Zig Implementation

```zig
pub fn apply_rope(
    q: []f32,           // [n_heads, head_dim]
    k: []f32,           // [n_kv_heads, head_dim]
    pos: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_theta: f32,
) void {
    const half = head_dim / 2;

    // Precompute frequency for this position
    var freqs: [32]f32 = undefined;  // max head_dim = 64
    for (0..half) |i| {
        const freq = rope_theta * @exp(-@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
        freqs[i] = pos * freq;
    }

    // Apply to query heads
    for (0..n_heads) |h| {
        const base = h * head_dim;
        for (0..half) |i| {
            const cos_val = @cos(freqs[i]);
            const sin_val = @sin(freqs[i]);

            const x0 = q[base + i];
            const x1 = q[base + i + half];

            q[base + i] = x0 * cos_val - x1 * sin_val;
            q[base + i + half] = x0 * sin_val + x1 * cos_val;
        }
    }

    // Apply to key heads
    for (0..n_kv_heads) |h| {
        const base = h * head_dim;
        for (0..half) |i| {
            const cos_val = @cos(freqs[i]);
            const sin_val = @sin(freqs[i]);

            const x0 = k[base + i];
            const x1 = k[base + i + half];

            k[base + i] = x0 * cos_val - x1 * sin_val;
            k[base + i + half] = x0 * sin_val + x1 * cos_val;
        }
    }
}
```

---

## 9. SwiGLU: Feed-Forward Network

### 9.1 SwiGLU Formula

```
SwiGLU(x) = SiLU(W_gate(x)) * W_up(x) * W_down
         = (x * sigmoid(x)) * up(x)
```

Where:
- gate_proj: [hidden_dim, dim]
- up_proj: [hidden_dim, dim]
- down_proj: [dim, hidden_dim]

### 9.2 Zig Implementation

```zig
fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn ffn_swiglu(
    output: []f32,       // [dim] - written to
    input: []const f32,  // [dim]
    model: *Model,
    layer: u32,
    ctx: *ScratchBuffers,
) void {
    const hidden_dim = model.spec.hidden_dim;
    const dim = model.spec.dim;

    const gate = ctx.ffn_temp[0..hidden_dim];
    const up = ctx.ffn_temp[hidden_dim..2 * hidden_dim];

    // gate = input @ gate_proj.T
    matmul(gate, input, model.gate_proj[layer], hidden_dim, dim);

    // up = input @ up_proj.T
    matmul(up, input, model.up_proj[layer], hidden_dim, dim);

    // SiLU activation on gate
    for (gate) |*g| {
        g.* = silu(g.*);
    }

    // Multiply: gate *= up (element-wise)
    for (gate, up) |*g, u| {
        g.* *= u;
    }

    // output = gate @ down_proj.T
    matmul(output, gate, model.down_proj[layer], dim, hidden_dim);
}
```

---

## 10. Quantization: Support for Q4_K, Q8_0

### 10.1 Q4_K Block Structure

```zig
pub const Q4KBlock = struct {
    scales: [8]f32,      // 8 scale factors per 32 elements
    zeros: [8]f32,       // 8 zero points
    data: [16]u8,        // 16 bytes = 32 nibbles (4-bit each)
};

// Each block: 32 elements, 4-bit quantized
// 16 bytes data + 8*4 bytes scales + 8*4 bytes zeros = 80 bytes per 32 elements
// Compression: 256 bytes -> 80 bytes = 3.2x
```

### 10.2 Quantized Matrix-Vector Multiply

```zig
pub fn matmul_q4k(
    output: []f32,
    input: []const f32,
    weight: []const Q4KBlock,
    rows: usize,
    cols: usize,
) void {
    const block_size = 32;
    const blocks_per_row = cols / block_size;

    for (row in 0..rows) |r| {
        var sum: f32 = 0;

        for (block in 0..blocks_per_row) |b| {
            const block_data = weight[r * blocks_per_row + block];
            const scale_base = b * block_size;
            const data_base = b * 16;

            for (elem in 0..block_size) |e| {
                // Extract 4-bit value
                const byte_idx = elem / 2;
                const is_low = elem % 2 == 0;
                const raw = (block_data.data[data_base + byte_idx] >> (4 * @intFromBool(is_low))) & 0x0F;

                // Dequantize
                const val = @as(f32, @intFromFloat(raw)) * block_data.scales[elem] + block_data.zeros[elem];

                // Multiply-accumulate
                sum += input[scale_base + elem] * val;
            }
        }

        output[r] = sum;
    }
}
```

---

## 11. Performance Targets: ~100 tok/s

### 11.1 Optimization Strategy

**Phase 1: Correctness (Target: 10 tok/s)**
- Implement all operations correctly in pure Zig
- Verify output matches reference (Ollama)
- No SIMD yet

**Phase 2: Vectorization (Target: 50 tok/s)**
- Add `@Vector` SIMD for matmul
- 8-wide float32 operations
- Unroll attention loops

**Phase 3: Cache Optimization (Target: 100 tok/s)**
- KV cache layout optimization
- Memory access pattern tuning
- Prefetch hints

### 11.2 SIMD Matmul Kernel

```zig
const SimdF32x8 = @Vector(8, f32);

pub fn matmul_simd(
    output: []f32,
    a: []const f32,  // [m, k] row-major
    b: []const f32,  // [k, n] column-major
    m: usize,
    n: usize,
    k: usize,
) void {
    // Process 8 output elements per iteration
    var i: usize = 0;
    while (i + 7 < m * n) : (i += 8) {
        const row = i / n;
        const col_start = i % n;

        var sum: SimdF32x8 = @splat(0);

        var j: usize = 0;
        while (j + 7 < k) : (j += 8) {
            // Load 8 elements from a (row, j..j+7)
            var a_vec: SimdF32x8 = undefined;
            for (0..8) |idx| {
                a_vec[idx] = a[row * k + j + idx];
            }

            // Load 8 elements from b (j..j+7, col_start)
            var b_vec: SimdF32x8 = undefined;
            for (0..8) |idx| {
                b_vec[idx] = b[(j + idx) * n + col_start];
            }

            sum += a_vec * b_vec;
        }

        // Handle remainder
        const remainder = k % 8;
        for (0..remainder) |idx| {
            sum[idx] += a[row * k + j + idx] * b[(j + idx) * n + col_start];
        }

        // Store result
        output[i..i+8] = sum;
    }
}
```

### 11.3 Benchmarking Infrastructure

```zig
pub const BenchmarkResult = struct {
    tok_per_sec: f32,
    total_tokens: u32,
    prefill_ms: u32,
    decode_ms: u32,
    peak_rss_mb: f32,
};

pub fn benchmark(model: *Model, prompt: []const u8, n_tokens: u32) !BenchmarkResult {
    var gen_state = try GenerationState.init(model);

    const start = std.time.milliTimestamp();
    try prefill(&gen_state, prompt);
    const prefill_end = std.time.milliTimestamp();

    var tokens = std.ArrayList(u32).init();
    while (tokens.items.len < n_tokens) {
        const tok = try decode_next(&gen_state);
        if (is_eos(tok)) break;
        try tokens.append(tok);
    }

    const end = std.time.milliTimestamp();

    return BenchmarkResult{
        .tok_per_sec = @as(f32, @floatFromInt(tokens.items.len)) /
                       (@as(f32, @floatFromInt(end - start)) / 1000.0),
        .total_tokens = @as(u32, @intCast(tokens.items.len)),
        .prefill_ms = @as(u32, @intCast(prefill_end - start)),
        .decode_ms = @as(u32, @intCast(end - prefill_end)),
        .peak_rss_mb = get_peak_rss_mb(),
    };
}
```

---

## 12. Interface: CLI, TUI, WebUI, API

### 12.1 CLI Interface

```zig
pub const CliArgs = struct {
    model_path: []const u8,
    prompt: ?[]const u8 = null,
    max_tokens: u32 = 50,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 0,
    repetition_penalty: f32 = 1.3,
    mode: enum { cli, tui, web } = .cli,
    web_port: u16 = 7331,
};

pub fn run_cli(args: CliArgs) !void {
    const model = try load_model(args.model_path);
    defer model.deinit();

    if (args.prompt) |prompt| {
        // Single prompt mode
        const result = try generate_text(model, prompt, args);
        std.debug.print("{s}", .{result});
    } else {
        // Interactive mode
        try interactive_chat(model, args);
    }
}

fn interactive_chat(model: *Model, args: CliArgs) !void {
    const stdin = std.io.getStdIn().reader();
    const stdout = std.io.getStdOut().writer();

    try stdout.print("SmolLM2 Chat (type 'quit' to exit)\n\n", .{});

    var history = std.ArrayList(ChatMessage).init();
    defer history.deinit();

    while (true) {
        try stdout.print("You: ", .{});
        const input = (try stdin.readUntilDelimiterOrEof('\n')).?;
        if (input.len == 0) continue;

        if (std.mem.eql(u8, input, "quit") or std.mem.eql(u8, input, "exit")) {
            break;
        }

        try history.append(.{ .role = "user", .content = input });
        const prompt = try format_chat_prompt(history.items);
        const response = try generate_text(model, prompt, args);

        // Extract assistant response (after <|im_start|>assistant\n)
        const assistant_msg = extract_assistant_response(response);
        try history.append(.{ .role = "assistant", .content = assistant_msg });

        try stdout.print("\nSmolLM2: {s}\n\n", .{assistant_msg});
    }
}
```

### 12.2 TUI Interface (Optional)

```zig
pub fn run_tui(model: *Model, args: CliArgs) !void {
    // Use Zig's async陳述式 library when available
    // For now, simplified terminal UI

    try terminal.hide_cursor();
    defer terminal.show_cursor();

    var history = std.ArrayList(u8).init();

    while (true) {
        try terminal.clear_screen();
        try terminal.move_cursor(1, 1);

        // Draw chat history
        try print_chat_history(history);

        // Draw input prompt
        try terminal.move_cursor(terminal.height(), 1);
        try stdout.print("> ", .{});

        const input = (try stdin.readLine()).?;
        if (input.len == 0) continue;

        if (std.mem.eql(u8, input, ":quit")) break;

        // Generate and stream response
        try stream_response(model, input, &history);
    }
}
```

### 12.3 Web/API Interface (Optional)

```zig
pub fn run_api(model: *Model, port: u16) !void {
    const server = try HttpServer.init(port);
    defer server.deinit();

    while (true) {
        const request = try server.accept();
        defer request.deinit();

        if (std.mem.eql(u8, request.method, "POST") and
            std.mem.eql(u8, request.path, "/v1/chat/completions")) {
            // Parse JSON body
            const body = try parse_chat_request(request.body);

            // Generate response
            const response = try chat_completions(model, body);

            // Send SSE response
            try request.send_sse(response);
        }
    }
}
```

---

## 13. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] File format parsing (.sm2, GGUF)
- [ ] Model weight loading and validation
- [ ] Tokenizer implementation (BPE)
- [ ] Basic inference (prefill + decode)

### Phase 2: Correctness
- [ ] RMSNorm implementation
- [ ] RoPE implementation
- [ ] Attention with KV cache
- [ ] SwiGLU FFN
- [ ] Sampling (greedy, temperature, top-p)

### Phase 3: Performance
- [ ] SIMD matmul kernels
- [ ] Loop unrolling
- [ ] Memory access optimization
- [ ] KV cache optimization

### Phase 4: Interfaces
- [ ] CLI interface
- [ ] TUI (optional)
- [ ] HTTP API (optional)

### Phase 5: Quantization
- [ ] Q8_0 support
- [ ] Q4_K support

---

## 14. Testing Strategy

### 14.1 Unit Tests
- F16 conversion
- RoPE correctness
- RMSNorm numerical stability
- BPE encoding/decoding round-trip

### 14.2 Integration Tests
- Model loading verification
- End-to-end generation matching Ollama output
- Memory leak detection

### 14.3 Performance Tests
- tok/s benchmark against C baseline
- Memory usage profiling

---

## 15. Success Criteria

| Metric | Target | Validation |
|--------|--------|------------|
| Correctness | Output matches Ollama for "hello" prompt | Exact token match |
| EOS Detection | Proper termination at <\|im_end\|> (token 2 or 128009) | No infinite loops |
| Performance | ~100 tok/s on target hardware | Benchmark measurement |
| Memory | < 512MB peak RSS | /usr/bin/time -v |

---

*End of specification*