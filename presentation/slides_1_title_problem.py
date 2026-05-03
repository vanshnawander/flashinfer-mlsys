"""
Part 1 — Title, Problem Statement, Hardware, MoE Architecture
Slides: 1-5
"""
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

from style import (
    new_prs, blank_slide, fill_bg, add_rect, add_textbox, add_title_bar,
    add_section_label, bullet_list,
    BG_DARK, BG_CARD, ACCENT1, ACCENT2, ACCENT3, WHITE, GREY_MID,
    RED_ERR, NVIDIA_GRN, SLIDE_W, SLIDE_H
)
from data import PROBLEM_DEF, HARDWARE, WORKLOADS


def slide_title(prs):
    """Slide 1 — title splash."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)

    # Full-width gradient band
    add_rect(s, 0, Inches(2.6), SLIDE_W, Inches(2.4),
             fill_color=RGBColor(0x13, 0x1A, 0x26))
    # Cyan accent bar top
    add_rect(s, 0, Inches(2.6), SLIDE_W, Inches(0.04), fill_color=ACCENT1)
    # Green accent bar bottom
    add_rect(s, 0, Inches(5.0), SLIDE_W, Inches(0.04), fill_color=ACCENT2)

    # Main title
    add_textbox(s, Inches(0.7), Inches(2.75), Inches(12), Inches(0.9),
                "Optimizing MoE FP8 Kernels for NVIDIA B200",
                font_size=Pt(38), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    add_textbox(s, Inches(0.7), Inches(3.7), Inches(12), Inches(0.5),
                "MLSys'26 Contest · FlashInfer-Bench Track · Submission 14",
                font_size=Pt(18), color=ACCENT1, align=PP_ALIGN.CENTER)

    # Benchmark name chip
    add_rect(s, Inches(3.2), Inches(4.3), Inches(6.9), Inches(0.46),
             fill_color=RGBColor(0x00, 0x35, 0x4A))
    add_textbox(s, Inches(3.2), Inches(4.32), Inches(6.9), Inches(0.42),
                "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
                font_size=Pt(10.5), color=ACCENT1, align=PP_ALIGN.CENTER)

    # Top-left tags
    add_textbox(s, Inches(0.3), Inches(0.15), Inches(4), Inches(0.35),
                "NVIDIA B200 (Blackwell) · SM 10.0 · HBM3e 8 TB/s",
                font_size=Pt(11), color=GREY_MID)

    # Author bottom
    add_textbox(s, Inches(0.3), Inches(6.9), Inches(10), Inches(0.4),
                "humming-bird  ·  April 2026",
                font_size=Pt(11), color=GREY_MID)

    # Right side stats column
    for i, (label, val) in enumerate([
        ("Peak Speedup", "7.4×"),
        ("Avg Speedup", "~2.1×"),
        ("Workloads", "19 / 19"),
    ]):
        bx = Inches(11.0)
        by = Inches(2.85) + i * Inches(0.7)
        add_rect(s, bx, by, Inches(2.1), Inches(0.55),
                 fill_color=RGBColor(0x00, 0x2A, 0x3A))
        add_textbox(s, bx + Inches(0.07), by + Inches(0.0), Inches(2.0), Inches(0.27),
                    label, font_size=Pt(9), color=GREY_MID)
        add_textbox(s, bx + Inches(0.07), by + Inches(0.25), Inches(2.0), Inches(0.3),
                    val, font_size=Pt(17), bold=True, color=ACCENT2)


def slide_problem_statement(prs):
    """Slide 2 — what is the problem."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "The Problem", subtitle="Mixture-of-Experts Inference at Scale")

    # Left card — what is MoE
    add_rect(s, Inches(0.3), Inches(1.4), Inches(6.0), Inches(5.7), fill_color=BG_CARD)
    add_textbox(s, Inches(0.5), Inches(1.5), Inches(5.6), Inches(0.4),
                "What is Mixture-of-Experts?", font_size=Pt(16), bold=True, color=ACCENT1)

    moe_lines = [
        "Each token is routed to K=8 experts out of 256 total",
        "Only 32 experts exist locally (others on other GPUs)",
        "Each expert: 2 GEMMs (gate+up proj, down proj)",
        "Weights stored in FP8 with 128-element block scales",
        "Output accumulated in FP32, cast to BF16",
        "",
        "Scale: DeepSeek-V3 style — H=7168, I=2048",
        "32 local experts × 2 × GEMM = 64 matrix multiplications",
        "With T=14107 tokens: billions of FLOPs per forward pass",
    ]
    bullet_list(s, moe_lines, Inches(0.5), Inches(2.0), Inches(5.6), Inches(4.8),
                font_size=Pt(13))

    # Right card — the challenge
    add_rect(s, Inches(6.6), Inches(1.4), Inches(6.4), Inches(5.7), fill_color=BG_CARD)
    add_textbox(s, Inches(6.8), Inches(1.5), Inches(6.0), Inches(0.4),
                "Optimization Challenges", font_size=Pt(16), bold=True, color=ACCENT3)

    challenges = [
        "Non-uniform batch sizes per expert (Tk ∈ [0, T×8/32])",
        "FP8 weights need dequant before compute",
        "Sequential expert loop = 32× kernel launch overhead",
        "3.5 GB FP32 weight materialization per forward pass",
        "SwiGLU activation between GEMM1 and GEMM2",
        "",
        "Memory wall: 112 MB/expert FP32 vs 28 MB FP8",
        "Small-T dominated by launch latency, not compute",
        "Large-T is memory-bandwidth bound (8 TB/s HBM3e)",
    ]
    bullet_list(s, challenges, Inches(6.8), Inches(2.0), Inches(6.0), Inches(4.8),
                font_size=Pt(13), color=ACCENT3)


def slide_hardware(prs):
    """Slide 3 — B200 hardware specs."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Target Hardware", subtitle="NVIDIA B200 · Blackwell Architecture · SM 10.0")

    specs = [
        ("Streaming Multiprocessors", "160 SMs"),
        ("Shared Memory / SM", "228 KB"),
        ("Registers / Thread", "255 (32-bit)"),
        ("L2 Cache", "126 MB"),
        ("HBM3e Bandwidth", "8 TB/s"),
        ("FP8 Tensor Core TFLOPS", "~4,500 TFLOPS"),
        ("BF16 Tensor Core TFLOPS", "~2,250 TFLOPS"),
        ("TF32 Tensor Core TFLOPS", "~1,125 TFLOPS"),
        ("FP32 CUDA Core TFLOPS", "~90 TFLOPS"),
    ]

    cols = 3
    rows = (len(specs) + cols - 1) // cols
    cell_w = Inches(4.1)
    cell_h = Inches(1.0)
    gap = Inches(0.12)

    for i, (label, val) in enumerate(specs):
        col = i % cols
        row = i // cols
        x = Inches(0.3) + col * (cell_w + gap)
        y = Inches(1.5) + row * (cell_h + gap)
        fill = RGBColor(0x0A, 0x1E, 0x30) if col == 0 else BG_CARD
        add_rect(s, x, y, cell_w, cell_h, fill_color=fill)
        add_textbox(s, x + Inches(0.12), y + Inches(0.07), cell_w - Inches(0.2),
                    Inches(0.35), label, font_size=Pt(11), color=GREY_MID)
        add_textbox(s, x + Inches(0.12), y + Inches(0.42), cell_w - Inches(0.2),
                    Inches(0.45), val, font_size=Pt(19), bold=True, color=ACCENT1)

    # Key insight box
    add_rect(s, Inches(0.3), Inches(5.15), Inches(12.7), Inches(2.0),
             fill_color=RGBColor(0x00, 0x2A, 0x1A))
    add_rect(s, Inches(0.3), Inches(5.15), Inches(0.06), Inches(2.0),
             fill_color=ACCENT2)
    add_textbox(s, Inches(0.55), Inches(5.22), Inches(12.2), Inches(0.35),
                "KEY INSIGHT FOR KERNEL DESIGN", font_size=Pt(11), bold=True,
                color=ACCENT2)
    add_textbox(s, Inches(0.55), Inches(5.58), Inches(12.2), Inches(1.45),
                "FP8 E4M3 (4 mantissa bits) → BF16 (8 mantissa bits) is a LOSSLESS "
                "upcast. Load FP8 weights directly into tensor cores with BF16 accumulation "
                "= 4× less memory bandwidth + 2× higher TFLOPS than TF32 baseline. "
                "Apply block scales post-dot in FP32 for zero precision loss.",
                font_size=Pt(13), color=WHITE, wrap=True)


def slide_moe_pipeline(prs):
    """Slide 4 — MoE forward pass pipeline diagram."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "MoE Forward Pass Pipeline",
                  subtitle="DeepSeek-V3 Style · FP8 Block-Scale Quantization")

    # Pipeline stages as flow boxes
    stages = [
        ("Routing", "sigmoid + group\ntop-2/4/8 + normalize", ACCENT1),
        ("Dispatch", "nonzero → argsort\n→ unique_consecutive", ACCENT1),
        ("FP8 Gather", "bulk index_select\nall experts at once", ACCENT2),
        ("GEMM1 + SwiGLU", "FP8→BF16 tile cast\ngated MLP up-proj", ACCENT3),
        ("GEMM2", "down-proj + scale\nrouting weight fused", ACCENT3),
        ("Scatter-Add", "index_add_ accum\nFP32 → BF16 out", ACCENT2),
    ]

    box_w = Inches(1.9)
    box_h = Inches(1.4)
    y_box = Inches(2.0)
    gap = Inches(0.14)
    total_w = len(stages) * box_w + (len(stages) - 1) * gap
    x_start = (SLIDE_W - total_w) / 2

    for i, (name, detail, color) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        # Arrow connector
        if i > 0:
            ax = x - gap
            add_textbox(s, ax - Inches(0.1), y_box + box_h / 2 - Inches(0.15),
                        gap + Inches(0.2), Inches(0.3), "→",
                        font_size=Pt(18), color=GREY_MID, align=PP_ALIGN.CENTER)
        # Box
        add_rect(s, x, y_box, box_w, box_h, fill_color=BG_CARD)
        add_rect(s, x, y_box, box_w, Inches(0.05), fill_color=color)
        add_textbox(s, x + Inches(0.1), y_box + Inches(0.1), box_w - Inches(0.2),
                    Inches(0.4), name, font_size=Pt(13), bold=True, color=color)
        add_textbox(s, x + Inches(0.1), y_box + Inches(0.52), box_w - Inches(0.2),
                    Inches(0.8), detail, font_size=Pt(10.5), color=WHITE, wrap=True)

    # Workload params table
    add_rect(s, Inches(0.3), Inches(3.7), Inches(12.7), Inches(3.5), fill_color=BG_CARD)
    add_textbox(s, Inches(0.5), Inches(3.78), Inches(8), Inches(0.35),
                "Problem Parameters", font_size=Pt(14), bold=True, color=ACCENT1)

    params = [
        [("Hidden Size (H)", "7,168"), ("Intermediate (I)", "2,048"),
         ("Total Experts (E)", "256"), ("Local Experts", "32")],
        [("Top-K Selected", "8 (per token)"), ("N Groups", "8"),
         ("TopK Groups", "4"), ("Block Size Q", "128")],
        [("Input dtype", "FP8 E4M3"), ("Weights dtype", "FP8 E4M3 + block scales"),
         ("Output dtype", "BF16"), ("Sequence range", "1 – 14,107 tokens")],
    ]
    for rr, row in enumerate(params):
        for cc, (lbl, val) in enumerate(row):
            px = Inches(0.5) + cc * Inches(3.15)
            py = Inches(4.2) + rr * Inches(0.9)
            add_textbox(s, px, py, Inches(3.0), Inches(0.28),
                        lbl, font_size=Pt(9.5), color=GREY_MID)
            add_textbox(s, px, py + Inches(0.28), Inches(3.0), Inches(0.45),
                        val, font_size=Pt(13.5), bold=True, color=WHITE)


def slide_workloads(prs):
    """Slide 5 — 19 workloads overview."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Benchmark Workloads",
                  subtitle="19 workloads — sequence lengths 1 to 14,107")

    add_textbox(s, Inches(0.3), Inches(1.35), Inches(12.7), Inches(0.3),
                "Each workload is a single forward pass with different T (token count). "
                "The reference implementation uses plain PyTorch; we must beat it on B200.",
                font_size=Pt(12), color=GREY_MID, wrap=True)

    # Draw all 19 workloads as horizontal bars
    seqlens = sorted(WORKLOADS, key=lambda w: w["seq_len"])
    max_sl = max(w["seq_len"] for w in seqlens)
    bar_h = Inches(0.27)
    bar_gap = Inches(0.04)
    bar_x = Inches(1.8)
    bar_max_w = Inches(8.5)
    label_x = Inches(0.3)
    ref_x = Inches(10.5)

    for i, w in enumerate(seqlens):
        y = Inches(1.8) + i * (bar_h + bar_gap)
        frac = w["seq_len"] / max_sl
        # Small-T blue, medium cyan, large green
        if w["seq_len"] <= 15:
            bar_color = RGBColor(0x00, 0x6A, 0xFF)
        elif w["seq_len"] <= 100:
            bar_color = ACCENT1
        else:
            bar_color = ACCENT2
        add_rect(s, bar_x, y, bar_max_w * frac, bar_h, fill_color=bar_color)
        add_textbox(s, label_x, y, Inches(1.45), bar_h,
                    f"T={w['seq_len']}", font_size=Pt(9.5), color=WHITE)
        add_textbox(s, ref_x, y, Inches(2.5), bar_h,
                    f"ref={w['ref_ms']:.1f} ms", font_size=Pt(9.5), color=GREY_MID)

    # Legend
    for i, (lbl, col) in enumerate([
        ("T ≤ 15 (small / decode)", RGBColor(0x00, 0x6A, 0xFF)),
        ("T 16–100 (medium / batched)", ACCENT1),
        ("T > 100 (large / prefill)", ACCENT2),
    ]):
        lx = Inches(0.3) + i * Inches(4.3)
        ly = Inches(7.1)
        add_rect(s, lx, ly + Inches(0.07), Inches(0.22), Inches(0.22), fill_color=col)
        add_textbox(s, lx + Inches(0.3), ly, Inches(3.9), Inches(0.35),
                    lbl, font_size=Pt(11), color=WHITE)


def build(prs):
    slide_title(prs)
    slide_problem_statement(prs)
    slide_hardware(prs)
    slide_moe_pipeline(prs)
    slide_workloads(prs)
    return prs


if __name__ == "__main__":
    prs = new_prs()
    build(prs)
    out = "moe_presentation_part1.pptx"
    prs.save(out)
    print(f"Saved: {out} ({prs.slides.__len__()} slides)")
