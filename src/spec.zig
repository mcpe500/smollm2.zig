//! Model specifications for SmolLM2 variants
//! Based on Hugging Face SmolLM2 architecture

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

pub const Spec135M = SmolLM2Spec{
    .n_layers = 30,
    .dim = 576,
    .hidden_dim = 1536,
    .n_heads = 9,
    .n_kv_heads = 3,
};

pub const Spec360M = SmolLM2Spec{
    .n_layers = 32,
    .dim = 960,
    .hidden_dim = 2560,
    .n_heads = 15,
    .n_kv_heads = 5,
};

pub const Spec1700M = SmolLM2Spec{
    .n_layers = 24,
    .dim = 2048,
    .hidden_dim = 8192,
    .n_heads = 32,
    .n_kv_heads = 32,
};

pub fn getSpec(variant_id: u32) SmolLM2Spec {
    return switch (variant_id) {
        135 => Spec135M,
        360 => Spec360M,
        1700 => Spec1700M,
        else => @panic("Unknown variant"),
    };
}
