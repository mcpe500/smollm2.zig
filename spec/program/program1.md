# program.md

This document defines the automated framework for porting, correcting, and optimizing the SmolLM2-135M inference engine from its broken C state (`smollm2.c`) into a high-performance, rock-solid pure Zig implementation (`smollm2.zig`).

---

## 1. Setup

To set up a new optimization and development run, work with the user to:

1. **Agree on a run tag**: Propose a tag based on the format `smollm-zig-<date>` (e.g., `smollm-zig-may30`). The branch `dev/<tag>` must not already exist.
2. **Create the branch**: Execute `git checkout -b dev/<tag>` from the master branch to isolate experimentation.
3. **Scan the in-scope files**:
* `build.zig` — Standard compilation profiles and optimization flags (`-Doptimize=ReleaseFast`).
* `src/main.zig` — Command Line Interface, interaction handlers, and top-level loops.
* `src/model.zig` — Network weights mapping, layers definition, and memory architecture.
* `src/tensor.zig` — Linear algebra foundations, matrix multiplication, SIMD vector primitives.
* `src/sampler.zig` — Softmax calculations, temperature configurations, Top-P, and End-Of-Sequence (EOS) tracking.
* `tools/convert.py` — Ingestion utility parsing Hugging Face / Ollama pretrained weights into native packed binaries.


4. **Initialize results.tsv**: Create an empty `results.tsv` file containing only the designated header row.

---

## 2. Post-Mortem: Correcting the C Implementation Failures

The reference C architecture contains critical bugs that the Zig migration must fundamentally prevent:

* **Infinite Generation Loops (EOS Ignored)**: The C version fails to handle or decode token boundaries correctly, resulting in continuous generation loops up to the maximum context size ($8191$ characters) for simple inputs like `"hello"`. The Zig code must explicitly track and evaluate sequence terminators (`<|im_end|>` or `<|endoftext|>`).
* **Unvectorized Operations**: Processing speed drops to an unacceptable **7.1 tok/s** due to poorly aligned loops and a lack of explicit hardware acceleration. The Zig version must leverage compile-time calculations (`comptime`) and native `@Vector` mechanics to achieve **~100 tok/s**.
* **Memory Invalidation**: C pointer casting and uncontrolled allocations cause buffer issues during autoregressive generation. Zig must enforce deterministic, type-safe, and clear allocation patterns.

---

## 3. Experimentation Strategy

The development pipeline moves progressively through four distinct phases:

```
[Phase 1: Ingestion Pipeline] ──► [Phase 2: Mathematical Correctness]
                                                 │
                                                 ▼
[Phase 4: Interface Assembly] ◄── [Phase 3: SIMD Engine Optimization]

```

### What You CAN Do:

* Optimize loops, memory indexing, caching configurations, and math routines inside the `src/` folder.
* Introduce custom Zig `@Vector` parallelizations to achieve maximum hardware utilization.
* Modify the prompt processing routines to match structural alignment rules.

### What You CANNOT Do:

* Add bloated external C dependencies or runtimes. The solution must remain direct, lightweight, and written in pure Zig.
* Remove structural verification checks or compromise generation correctness just to boost raw token generation numbers.

### Evaluation Criteria:

1. **Intelligence Match**: The generated text must precisely match the intelligence and intent seen in reference implementations like Ollama. If the user prompts `"hello"`, the model should respond with a crisp, human-like greeting without rambling.
2. **Performance Target**: Achieve execution speeds approaching **~100 tokens/second** on target consumer configurations.

---

## 4. Output Format

Upon finalizing an inference run or a batch simulation test, output a telemetry log structured exactly like this:

```
---
chat_valid:       SUCCESS
tokens_per_sec:   104.2
total_tokens:     18
response_sample:  "Hello! I am SmolLM2, a helpful AI assistant. How can I assist you today?"
peak_rss_mb:      240.5
execution_status: CLEAN_TERMINATION

```

If a run fails or enters an unconstrained loop, output the following:

```
---
chat_valid:       FAILED_LOOP
tokens_per_sec:   7.4
total_tokens:     2048
response_sample:  "I am trying to figure out how many days 2018 is from this year. I was... [truncated]"
peak_rss_mb:      512.0
execution_status: CONTEXT_EXHAUSTION

```

---

## 5. Logging Results

Record your experiments in `results.tsv` using a tab-separated values format. Do not use commas, as they will corrupt descriptions.

### TSV Columns:

1. `commit`: Short Git commit hash (7 characters).
2. `tok_per_sec`: Numerical tokens per second output rate.
3. `chat_status`: Categorized outcome (`pass`, `loop_fail`, or `crash`).
4. `status`: Action decision taken on the iteration (`keep` or `discard`).
5. `description`: Concise documentation outlining the algorithmic change.

### Example Logs:

```
commit	tok_per_sec	chat_status	status	description
e1a2f3b	8.2	loop_fail	discard	initial naive C-to-Zig structural rewrite
a4b5c6d	9.5	pass	keep	fix EOS token mapping logic to prevent infinite text loops
f7g8h9i	68.4	pass	keep	vectorize matrix-vector multiplication loops via comptime @Vector
b2c3d4e	102.1	pass	keep	unroll attention layers and cache-align key-value pointers

```

---

## 6. The Experiment Loop

Execute this loop continuously and autonomously:

1. Inspect the state of the active development branch.
2. Code your structural optimizations or bug fixes directly into the codebase.
3. Commit your localized changes to Git.
4. Execute the verification pipeline using `-Doptimize=ReleaseFast`. Capture the console trace and filter metrics into a local log file:
```bash
zig build run --release=fast > run.log 2>&1

```


5. Extract status tags, accuracy checks, and performance numbers from the execution logs.
6. If the run encounters a compilation bug or crashes, review the standard error trace to fix typos or pointer errors. Skip any fundamentally flawed optimization concepts.
7. Append metrics into `results.tsv` (keep the file untracked by Git).
8. **Decide Branch Advancement**:
* If `chat_status` passes AND `tok_per_sec` increases, **advance** the development line and retain the commit.
* If performance degrades, output matches the old C generation loop error, or text outputs show errors, discard the change and run `git reset --hard HEAD~1`.



**NEVER STOP**: Maintain complete autonomy during this process. Do not pause to ask for human confirmation or input. If you run out of ideas, analyze the attention matrices, adjust loop bounds, look for micro-optimization wins in tensor structures, or optimize memory access layers. Run continuously until manually stopped by the user.
