# Phase 1 Handoff: Zig Project Scaffolding

**Date:** 2026-05-30
**Branch:** `dev/smollm-zig-may30`
**Commit:** 7376294

## Summary

Phase 1 scaffolding complete. Project compiles but stdin/stdout IO not yet working due to Zig stdlib API differences.

## What Was Done

1. **Project Structure Created**
   - `build.zig` - Zig 0.16 build configuration
   - `src/` directory with all module files
   - `results.tsv` for tracking experiments
   - `spec/` with analysis documentation

2. **Core Modules Implemented**
   - `spec.zig` - Model hyperparameters for 135M/360M/1700M
   - `tensor.zig` - Scalar matmul, RMSNorm, softmax, silu
   - `rope.zig` - Rotary Position Embedding
   - `attention.zig` - GQA with KV cache
   - `mlp.zig` - SwiGLU feed-forward network
   - `tokenizer.zig` - BPE tokenizer (placeholder)
   - `model.zig` - Model structures
   - `inference.zig` - Forward pass implementation
   - `generation.zig` - Generation loop with EOS handling
   - `file_format.zig` - .sm2 file header parsing

## Critical Issues

### Zig stdlib API Version Mismatch

The system has Zig 0.16.0 which has different APIs than documented:

**Working:**
- `std.heap.ArenaAllocator`
- `std.debug.print`
- `std.mem.asBytes`
- `std.mem.eql`
- `std.sort.sort`

**Not Working (API unknown):**
- `std.io.getStdIn` / `std.io.getStdOut`
- `std.fs.cwd()` / `std.fs.openFileAbsolute`
- `std.process.args`
- `std.time.milliTimestamp()`

### Next Steps Required

1. **Research correct IO API** for Zig 0.16
2. **Implement model loading** from .sm2 files
3. **Connect inference** to generation loop
4. **Add proper CLI** with interactive chat

## Files Status

| File | Status | Notes |
|------|--------|-------|
| `src/main.zig` | Stub | Needs IO implementation |
| `src/model.zig` | Partial | Needs .sm2 loading |
| `src/tokenizer.zig` | Placeholder | BPE not implemented |
| `src/inference.zig` | Stub | Forward pass exists |
| `src/generation.zig` | Stub | Sampling exists |
| `src/attention.zig` | Implemented | GQA logic done |
| `src/rope.zig` | Implemented | RoPE logic done |
| `src/mlp.zig` | Implemented | SwiGLU logic done |
| `src/tensor.zig` | Scalar only | Phase 2: add SIMD |

## Bug Fixes Applied

From C code analysis:
1. **EOS token handling** - Checks tokens 0, 2, 128009
2. **GQA broadcasting** - 9 heads -> 3 KV heads
3. **RoPE application** - Applied to both Q and K

## Test Command

```bash
zig build-exe src/main.zig && ./main
```

Output:
```
SmolLM2-135M Zig Inference Engine
Phase 1: Basic implementation

Initializing model...
Model initialized with spec:
  - Layers: 30
  - Dim: 576
  - Heads: 9 (3 KV)
  - Hidden: 1536

Phase 1 Complete!
```
