"""
Part 2 — Solution Approaches in Detail
Slides: 6-12  (one approach per submission cluster)
"""
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

from style import (
    new_prs, blank_slide, fill_bg, add_rect, add_textbox, add_title_bar,
    add_section_label, bullet_list,
    BG_DARK, BG_CARD, ACCENT1, ACCENT2, ACCENT3, WHITE, GREY_MID, RED_ERR, SLIDE_W, SLIDE_H
)
from data import SUBMISSIONS


def _approach_slide(prs, title, subtitle, tag, left_title, left_bullets,
                    right_title, right_bullets, code_snippet=None,
                    left_color=ACCENT1, right_color=ACCENT2):
    """Template: left = what changed, right = result/impact, optional code box."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, title, subtitle=subtitle)

    # Tag chip
    add_rect(s, SLIDE_W - Inches(2.6), Inches(0.22), Inches(2.3), Inches(0.38),
             fill_color=RGBColor(0x00, 0x35, 0x4A))
    add_textbox(s, SLIDE_W - Inches(2.6), Inches(0.25), Inches(2.3), Inches(0.32),
                tag, font_size=Pt(11), bold=True, color=ACCENT1, align=PP_ALIGN.CENTER)

    split = Inches(6.5) if code_snippet is None else Inches(6.5)

    # Left panel
    add_rect(s, Inches(0.3), Inches(1.35), split - Inches(0.45), Inches(6.0),
             fill_color=BG_CARD)
    add_textbox(s, Inches(0.5), Inches(1.42), split - Inches(0.65), Inches(0.38),
                left_title, font_size=Pt(14), bold=True, color=left_color)
    bullet_list(s, left_bullets,
                Inches(0.5), Inches(1.9), split - Inches(0.65), Inches(5.2),
                font_size=Pt(12.5), color=WHITE, bullet_color=left_color)

    # Right panel
    right_x = split + Inches(0.1)
    right_w = SLIDE_W - right_x - Inches(0.3)

    if code_snippet:
        # Code box takes bottom 40%, right panel at top
        right_h = Inches(2.6)
        add_rect(s, right_x, Inches(1.35), right_w, right_h, fill_color=BG_CARD)
        add_textbox(s, right_x + Inches(0.15), Inches(1.42), right_w - Inches(0.3),
                    Inches(0.38), right_title, font_size=Pt(14), bold=True,
                    color=right_color)
        bullet_list(s, right_bullets,
                    right_x + Inches(0.15), Inches(1.9), right_w - Inches(0.3),
                    Inches(1.9), font_size=Pt(12), color=WHITE, bullet_color=right_color)

        # Code box
        code_y = Inches(4.1)
        add_rect(s, right_x, code_y, right_w, Inches(3.25),
                 fill_color=RGBColor(0x08, 0x12, 0x1A))
        add_rect(s, right_x, code_y, right_w, Inches(0.04), fill_color=ACCENT1)
        add_textbox(s, right_x + Inches(0.1), code_y + Inches(0.08),
                    right_w - Inches(0.2), Inches(0.28),
                    "Key Code", font_size=Pt(9), bold=True, color=ACCENT1)
        add_textbox(s, right_x + Inches(0.1), code_y + Inches(0.4),
                    right_w - Inches(0.2), Inches(2.7),
                    code_snippet, font_size=Pt(9), color=RGBColor(0xA8, 0xD8, 0xFF),
                    wrap=True)
    else:
        add_rect(s, right_x, Inches(1.35), right_w, Inches(6.0), fill_color=BG_CARD)
        add_textbox(s, right_x + Inches(0.15), Inches(1.42), right_w - Inches(0.3),
                    Inches(0.38), right_title, font_size=Pt(14), bold=True,
                    color=right_color)
        bullet_list(s, right_bullets,
                    right_x + Inches(0.15), Inches(1.9), right_w - Inches(0.3),
                    Inches(5.2), font_size=Pt(12.5), color=WHITE,
                    bullet_color=right_color)


def slide_sub1_baseline(prs):
    _approach_slide(
        prs,
        title="Sub-1: Baseline Correctness",
        subtitle="Get the math right before optimizing · 19/19 PASSED",
        tag="sub-1 · avg 1.74×",
        left_title="Bugs Fixed",
        left_bullets=[
            "DPS signature: framework passes output as 11th arg",
            "  kernel only accepted 10 → TypeError crash",
            "Fix: add `output: torch.Tensor` param, write in-place",
            "",
            "Routing logic must match reference exactly:",
            "  sigmoid(logits) + bias → group view(T,8,32)",
            "  topk(k=2, dim=2) sum → group scores",
            "  topk(k=4, dim=1) → group mask → expand",
            "  topk(k=8, pruned) → global expert selection",
            "  gather(s, topk_idx) / sum * scaling_factor",
            "",
            "FP8 block-scale dequant: view(n,128,m,128) × scale",
            "SwiGLU: gate × silu(up) in FP32 accumulation",
        ],
        right_title="Lesson Learned",
        right_bullets=[
            "Read the framework source before writing one line",
            "DPS = Destination Passing Style — pre-allocated output",
            "  The framework controls output memory lifecycle",
            "",
            "Routing is the critical path to match exactly:",
            "  Off-by-one in group_mask → wrong experts selected",
            "  Wrong top-k order → different token weights",
            "",
            "Baseline gives 2.86–4.35× on small-T (T=1,7)",
            "Large-T (T=14107) only 1.03× — huge room to improve",
            "",
            "All 19 workloads PASSED — correctness is the floor",
        ],
        code_snippet=(
            "# FP8 block-scale dequant\n"
            "x = hidden.to(fp32).view(T, H//128, 128)\n"
            "s = scale.t().unsqueeze(2)  # (T, H//128, 1)\n"
            "a_fp32 = (x * s).reshape(T, H)\n\n"
            "# Routing\n"
            "s_wb = sigmoid(logits) + bias     # (T, 256)\n"
            "groups = s_wb.view(T, 8, 32)      # group view\n"
            "g_scores = topk(groups,k=2).sum() # sum top-2\n"
            "mask = scatter topk(g_scores,k=4) # 4 groups\n"
            "topk_idx = topk(masked, k=8)      # final 8"
        ),
        left_color=ACCENT3, right_color=ACCENT1
    )


def slide_sub3_dispatch(prs):
    _approach_slide(
        prs,
        title="Sub-3: Dispatch Table — 96 ops → 4",
        subtitle="Biggest single improvement · eliminated Python expert loop overhead",
        tag="sub-3 · avg 2.17×",
        left_title="What Changed",
        left_bullets=[
            "BEFORE: per-expert loop with .nonzero() + .any()",
            "  for exp in range(32):",
            "    mask = (topk_idx == exp)  # launch!",
            "    if mask.any():             # launch!",
            "      tokens = mask.nonzero() # launch!",
            "  → 32 × 3 = 96 kernel launches",
            "",
            "AFTER: vectorized dispatch table",
            "  all_valid = nonzero(valid_local)  # 1 launch",
            "  sort_order = argsort(expert_ids)  # 1 launch",
            "  unique, counts = unique_consecutive # 1 launch",
            "  boundaries = cumsum(counts)        # 1 launch",
            "  → loop reads CPU slices, no GPU ops",
        ],
        right_title="Impact",
        right_bullets=[
            "Small-T (T=1): 4.35× → 6.50× (+49%)",
            "Small-T (T=7): 2.86× → 3.73× (+30%)",
            "Mid-T (T=52): 1.60× → 1.85× (+16%)",
            "Large-T (T=14107): 1.03→1.85× (+80%!)",
            "",
            "Sub-3 is the single largest improvement across ALL",
            "workload sizes — dispatch was ~60% of runtime on",
            "small-T sequences",
            "",
            "Average speedup: 1.74× → 2.17× (+25%)",
            "",
            "Key: stable argsort preserves token ordering",
            "within each expert → numerically deterministic",
        ],
        left_color=ACCENT1, right_color=ACCENT2
    )


def slide_sub9_bf16_tensorcores(prs):
    _approach_slide(
        prs,
        title="Sub-9: BF16 Tensor Cores via Lossless FP8→BF16 Cast",
        subtitle="Key insight: FP8 E4M3 is a subset of BF16 — cast is lossless",
        tag="sub-9 · avg 2.23×",
        left_title="The Core Insight",
        left_bullets=[
            "FP8 E4M3: sign=1, exp=4, mantissa=3 bits",
            "BF16:     sign=1, exp=8, mantissa=7 bits",
            "→ FP8→BF16 upcast is LOSSLESS (all values preserved)",
            "",
            "Before: load FP8 → dequant to FP32 → TF32 matmul",
            "  Bandwidth: 4 B/elem (FP32 tiles), 60 TFLOPS",
            "",
            "After:  load FP8 → cast to BF16 tile → BF16 matmul",
            "  Bandwidth: 1 B/elem (FP8 tiles), 2,250 TFLOPS",
            "  Apply block scale after dot in FP32 (no precision loss)",
            "",
            "Triton kernel: load FP8 tile, .to(tl.bfloat16),",
            "  tl.dot(a_tile, tl.trans(w_tile)) → FP32 accumulator",
            "  multiply by a_scale × w_scale post-dot",
        ],
        right_title="Results",
        right_bullets=[
            "T=1: 1.390 ms → peak 7.43× speedup",
            "T=14107: 18.540 ms (2.35× vs ref 45.4 ms)",
            "T=11948: 14.886 ms (2.32× vs ref 34.5 ms)",
            "",
            "Eliminated separate FP32 weight materialization:",
            "  Saved 112 MB/expert × 32 = 3.5 GB per pass",
            "",
            "Triton SMEM budget (safe < 228 KB):",
            "  A-tile: 64×128×1 B = 8 KB per stage",
            "  W-tile: 128×128×1 B = 16 KB per stage",
            "  3 stages × 24 KB = 72 KB — well within budget",
            "",
            "num_stages=3 for GEMM1 (56 K-iters at K=7168)",
            "Bulk FP8 gather: index_select once for all experts",
        ],
        code_snippet=(
            "# Load FP8 tile, cast to BF16 — LOSSLESS\n"
            "a_tile = tl.load(a_ptr + ...).to(tl.bfloat16)\n"
            "w_gate = tl.load(w_ptr + ...).to(tl.bfloat16)\n"
            "w_up   = tl.load(w_ptr + ...).to(tl.bfloat16)\n\n"
            "# BF16 tensor cores: 2,250 TFLOPS!\n"
            "raw_gate = tl.dot(a_tile, tl.trans(w_gate))\n"
            "raw_up   = tl.dot(a_tile, tl.trans(w_up))\n\n"
            "# Post-dot scale in FP32 (zero precision loss)\n"
            "a_s  = tl.load(a_scale_ptr + k_blk*... + m*...)\n"
            "gate_acc += raw_gate * (a_s[:,None] * ws_gate)"
        ),
        left_color=ACCENT2, right_color=ACCENT1
    )


def slide_sub13_epilogue(prs):
    _approach_slide(
        prs,
        title="Sub-13: Route-Weight Fused into GEMM2 Epilogue",
        subtitle="3 lines of code · saves one full scatter pass per expert",
        tag="sub-13 · avg 2.11×",
        left_title="What Changed",
        left_bullets=[
            "BEFORE (sub-9 GEMM2): output to o_buf then multiply",
            "  o_buf[i] = gemm2_result[i]",
            "  o_buf *= route_w[:, None]          # extra pass!",
            "  accum.index_add_(0, t_idx, o_buf)  # scatter",
            "",
            "AFTER: multiply in Triton epilogue",
            "  route_w = tl.load(route_w_ptr + offs_m)",
            "  acc = acc * route_w[:, None]  # in-register!",
            "  tl.store(o_ptr + ...)",
            "",
            "Memory saved per expert per forward pass:",
            "  Tk × 7168 × 4 B read + Tk × 7168 × 4 B write",
            "  = 2 × Tk × 28 KB eliminated from HBM traffic",
            "",
            "Safe: zero SMEM impact (route_w fits in 1 register)",
        ],
        right_title="Design Principle: In-Register Fusion",
        right_bullets=[
            "The key insight: data in registers costs NOTHING",
            "  Registers are on-chip, 0 bandwidth cost",
            "  Any multiply in the epilogue is 'free' compute",
            "",
            "Pattern applicable whenever you have:",
            "  1. A per-row scalar weighting",
            "  2. Applied immediately after matmul",
            "  → Always fuse into epilogue, never a separate pass",
            "",
            "Sub-13 is conservative: only sub-9 + epilogue fusion",
            "  No risky SMEM changes, no tile-size experiments",
            "  Stable 19/19 PASSED across all B200 runs",
            "",
            "Combined with sub-9: best overall Triton result",
        ],
        left_color=ACCENT1, right_color=ACCENT3
    )


def slide_sub_cuda(prs):
    """CUDA ATen C++ approach slide."""
    _approach_slide(
        prs,
        title="CUDA Path: ATen C++ with Custom Kernels",
        subtitle="Pure CUDA C++ — no Python overhead, custom fused ops",
        tag="cuda-16",
        left_title="Architecture",
        left_bullets=[
            "Language: CUDA C++ with PyTorch TorchBuilder",
            "Entry: kernel.cu::kernel (DPS style, 11 params)",
            "Build: torch.utils.cpp_extension.load() + nvcc",
            "",
            "Custom CUDA kernels (no ATen dispatch overhead):",
            "  fused_sigmoid_bias: __expf intrinsic,1 kernel pass",
            "  swiglu_kernel: gate×silu(up) in single kernel",
            "  weighted_scatter_add: atomicAdd-based fusion",
            "",
            "GEMMs via cuBLAS (at::matmul) with:",
            "  setFloat32MatmulPrecision('highest') — no TF32",
            "  Pre-gathered tokens: batch index_select upfront",
            "",
            "Environment fix: add cuda-nvcc-12-8 to Modal image",
            "3 bugs fixed: no nvcc, wrong binding, wrong entry pt",
        ],
        right_title="Triton vs CUDA (head-to-head)",
        right_bullets=[
            "Triton wins on ALL 19/19 workloads:",
            "  T=1:     Triton 1.39 ms  vs CUDA 1.58 ms  (+14%)",
            "  T=7:     Triton 2.84 ms  vs CUDA 3.06 ms  (+8%)",
            "  T=901:   Triton 12.4 ms  vs CUDA 18.5 ms  (+49%)",
            "  T=14107: Triton 18.5 ms  vs CUDA 42.0 ms  (+127%!)",
            "",
            "Why Triton dominates at large-T:",
            "  FP8 on-the-fly dequant in K-loop = 4× less BW",
            "  BF16 tensor cores vs TF32 = 2× TFLOPS",
            "  CUDA path materializes full FP32 weight matrices",
            "",
            "CUDA advantage: better numerical stability",
            "  Large-T rel_err: CUDA 0.5–5  vs Triton 10⁹",
            "  TF32 disabled (FP32 matmul) = exact accumulation",
        ],
        left_color=ACCENT3, right_color=ACCENT2
    )


def build(prs):
    slide_sub1_baseline(prs)
    slide_sub3_dispatch(prs)
    slide_sub9_bf16_tensorcores(prs)
    slide_sub13_epilogue(prs)
    slide_sub_cuda(prs)
    return prs


if __name__ == "__main__":
    prs = new_prs()
    build(prs)
    out = "moe_presentation_part2.pptx"
    prs.save(out)
    print(f"Saved: {out} ({prs.slides.__len__()} slides)")
