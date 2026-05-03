"""
Part 3 — Results, Performance Analysis, Lessons Learned
Slides: 13-18
"""
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

from style import (
    new_prs, blank_slide, fill_bg, add_rect, add_textbox, add_title_bar,
    add_section_label, bullet_list, speedup_color,
    BG_DARK, BG_CARD, ACCENT1, ACCENT2, ACCENT3, WHITE, GREY_MID, RED_ERR,
    SLIDE_W, SLIDE_H
)
from data import SUBMISSIONS, WORKLOADS, BOTTLENECK_ANALYSIS, OPTIMIZATIONS, NEXT_STEPS


# Helpers
def _bar(slide, x, y, w, h, fill, label=None, label_color=WHITE, font_size=Pt(9)):
    add_rect(slide, x, y, max(w, Inches(0.01)), h, fill_color=fill)
    if label:
        add_textbox(slide, x + Inches(0.03), y, Inches(0.8), h,
                    label, font_size=font_size, color=label_color)


def slide_speedup_journey(prs):
    """Slide: per-submission average speedup journey (bar chart)."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Speedup Journey — Submission by Submission",
                  subtitle="Average speedup across all 19 workloads · B200")

    subs = ["sub-1", "sub-2", "sub-3", "sub-5", "sub-6", "sub-9", "sub-12", "sub-13", "sub-14"]
    avgs = [SUBMISSIONS[k]["avg_speedup"] for k in subs]
    labels = ["Sub-1", "Sub-2", "Sub-3", "Sub-5", "Sub-6", "Sub-9", "Sub-12", "Sub-13", "Sub-14"]
    changes = [
        "Baseline",
        "+fused GEMM",
        "+dispatch table",
        "+cuBLAS TF32",
        "+reverted fix",
        "+BF16 TC",
        "SMEM OOM",
        "+epilogue fuse",
        "clean rerun",
    ]

    max_speedup = 2.5
    chart_x = Inches(0.6)
    chart_y_top = Inches(1.5)
    chart_h = Inches(4.5)
    chart_w = SLIDE_W - Inches(1.2)
    bar_w = chart_w / len(subs) - Inches(0.18)

    # Grid lines
    for grid_val in [1.0, 1.5, 2.0, 2.5]:
        gy = chart_y_top + chart_h * (1 - grid_val / max_speedup)
        add_rect(s, chart_x, gy, chart_w, Inches(0.01),
                 fill_color=RGBColor(0x28, 0x38, 0x48))
        add_textbox(s, chart_x - Inches(0.45), gy - Inches(0.12),
                    Inches(0.4), Inches(0.3),
                    f"{grid_val:.1f}×", font_size=Pt(9), color=GREY_MID,
                    align=PP_ALIGN.RIGHT)

    for i, (sub, avg, lbl, ch) in enumerate(zip(subs, avgs, labels, changes)):
        bx = chart_x + i * (chart_w / len(subs)) + Inches(0.09)
        bh = chart_h * (avg / max_speedup)
        by = chart_y_top + chart_h - bh
        col = speedup_color(avg)
        if sub == "sub-12":  # regression
            col = RED_ERR
        _bar(s, bx, by, bar_w, bh, fill=col)
        # Value label on top
        add_textbox(s, bx, by - Inches(0.28), bar_w, Inches(0.28),
                    f"{avg:.2f}×", font_size=Pt(10), bold=True,
                    color=col, align=PP_ALIGN.CENTER)
        # Sub label below
        add_textbox(s, bx, chart_y_top + chart_h + Inches(0.05),
                    bar_w, Inches(0.28),
                    lbl, font_size=Pt(9.5), color=WHITE, align=PP_ALIGN.CENTER)
        # Change label (rotated via wrapping)
        add_textbox(s, bx - Inches(0.05), chart_y_top + chart_h + Inches(0.35),
                    bar_w + Inches(0.1), Inches(0.65),
                    ch, font_size=Pt(8), color=GREY_MID,
                    align=PP_ALIGN.CENTER, wrap=True)

    # Best result callout
    add_rect(s, Inches(9.5), Inches(1.4), Inches(3.5), Inches(1.6),
             fill_color=RGBColor(0x00, 0x2A, 0x1A))
    add_rect(s, Inches(9.5), Inches(1.4), Inches(0.06), Inches(1.6),
             fill_color=ACCENT2)
    add_textbox(s, Inches(9.65), Inches(1.48), Inches(3.2), Inches(0.3),
                "BEST RESULT", font_size=Pt(10), bold=True, color=ACCENT2)
    add_textbox(s, Inches(9.65), Inches(1.78), Inches(3.2), Inches(0.55),
                "Sub-9: avg 2.23×\nPeak: 7.43× (T=1)",
                font_size=Pt(14), bold=True, color=WHITE)


def slide_per_workload_table(prs):
    """Slide: best submission results per workload."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Sub-9 Results — All 19 Workloads",
                  subtitle="Best overall Triton submission · 19/19 PASSED")

    best = SUBMISSIONS["sub-9"]
    speedups = best["speedups"]

    # Table header
    headers = ["Seq Len", "Ref (ms)", "Our (ms)", "Speedup", "Status"]
    col_ws = [Inches(1.4), Inches(1.4), Inches(1.4), Inches(1.6), Inches(1.4)]
    hdr_y = Inches(1.45)
    hdr_x = Inches(0.3)
    add_rect(s, hdr_x, hdr_y, SLIDE_W - Inches(0.6), Inches(0.32),
             fill_color=RGBColor(0x0A, 0x22, 0x38))
    x = hdr_x + Inches(0.1)
    for h, cw in zip(headers, col_ws):
        add_textbox(s, x, hdr_y + Inches(0.03), cw, Inches(0.26),
                    h, font_size=Pt(10), bold=True, color=ACCENT1)
        x += cw

    # Rows
    wl_sorted = sorted(WORKLOADS, key=lambda w: w["seq_len"])
    row_h = Inches(0.3)
    for ri, w in enumerate(wl_sorted):
        idx_in_orig = next(j for j, ow in enumerate(WORKLOADS) if ow["uuid"] == w["uuid"])
        sp = speedups[idx_in_orig]
        our_ms = w["ref_ms"] / sp if sp else None
        ry = hdr_y + Inches(0.32) + ri * row_h
        row_bg = RGBColor(0x12, 0x1C, 0x28) if ri % 2 == 0 else BG_DARK
        add_rect(s, hdr_x, ry, SLIDE_W - Inches(0.6), row_h, fill_color=row_bg)
        col = speedup_color(sp)
        vals = [
            f"T={w['seq_len']}",
            f"{w['ref_ms']:.2f}",
            f"{our_ms:.2f}" if our_ms else "FAILED",
            f"{sp:.2f}×" if sp else "—",
            "✓ PASSED" if sp else "✗ ERROR",
        ]
        x = hdr_x + Inches(0.1)
        for vi, (val, cw) in enumerate(zip(vals, col_ws)):
            vc = col if vi >= 3 else WHITE
            if vi == 4 and sp:
                vc = ACCENT2
            add_textbox(s, x, ry + Inches(0.03), cw, Inches(0.26),
                        val, font_size=Pt(10), color=vc)
            x += cw


def slide_small_vs_large(prs):
    """Slide: bottleneck analysis by workload size."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Bottleneck Analysis by Workload Size",
                  subtitle="Different regimes have fundamentally different bottlenecks")

    categories = [
        ("small_T", ACCENT1, Inches(0.3)),
        ("medium_T", ACCENT3, Inches(4.5)),
        ("large_T", ACCENT2, Inches(8.7)),
    ]

    for key, color, cx in categories:
        bdata = BOTTLENECK_ANALYSIS[key]
        cw = Inches(4.0)
        card_y = Inches(1.45)
        card_h = Inches(5.8)
        add_rect(s, cx, card_y, cw, card_h, fill_color=BG_CARD)
        add_rect(s, cx, card_y, cw, Inches(0.05), fill_color=color)
        add_textbox(s, cx + Inches(0.15), card_y + Inches(0.1),
                    cw - Inches(0.3), Inches(0.38),
                    bdata["description"], font_size=Pt(15), bold=True, color=color)
        add_textbox(s, cx + Inches(0.15), card_y + Inches(0.52),
                    cw - Inches(0.3), Inches(0.28),
                    "Dominant bottleneck:", font_size=Pt(10), color=GREY_MID)
        add_textbox(s, cx + Inches(0.15), card_y + Inches(0.8),
                    cw - Inches(0.3), Inches(0.36),
                    bdata["dominant"], font_size=Pt(13), bold=True, color=WHITE)

        # Breakdown bars
        by = card_y + Inches(1.25)
        bar_max = cw - Inches(0.3)
        for comp, pct_str in bdata["breakdown"].items():
            raw = pct_str.replace("~", "").split("%")[0].strip()
            pct = float(raw) / 100
            lbl = comp.replace("_", " ").title()
            add_textbox(s, cx + Inches(0.15), by, cw - Inches(0.3), Inches(0.22),
                        lbl, font_size=Pt(9.5), color=GREY_MID)
            add_rect(s, cx + Inches(0.15), by + Inches(0.22),
                     bar_max * pct, Inches(0.18), fill_color=color)
            add_rect(s, cx + Inches(0.15), by + Inches(0.22),
                     bar_max, Inches(0.18),
                     fill_color=RGBColor(0x22, 0x30, 0x40))
            add_rect(s, cx + Inches(0.15), by + Inches(0.22),
                     bar_max * pct, Inches(0.18), fill_color=color)
            add_textbox(s, cx + Inches(0.15) + bar_max + Inches(0.05),
                        by + Inches(0.18), Inches(0.5), Inches(0.22),
                        pct_str, font_size=Pt(9), color=color)
            by += Inches(0.48)

        # Best fix
        add_rect(s, cx + Inches(0.1), by + Inches(0.1),
                 cw - Inches(0.2), Inches(0.04), fill_color=color)
        add_textbox(s, cx + Inches(0.15), by + Inches(0.2),
                    cw - Inches(0.3), Inches(0.26),
                    "Best fix:", font_size=Pt(9.5), bold=True, color=color)
        add_textbox(s, cx + Inches(0.15), by + Inches(0.46),
                    cw - Inches(0.3), Inches(1.0),
                    bdata["best_fix"], font_size=Pt(10), color=WHITE, wrap=True)


def slide_optimization_taxonomy(prs):
    """Slide: all optimizations organized by category."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Optimization Taxonomy",
                  subtitle="All techniques applied — categorized by type")

    colors = [ACCENT1, ACCENT3, ACCENT2, RGBColor(0xFF, 0xC4, 0x00), ACCENT1]
    col_w = Inches(2.48)
    col_gap = Inches(0.12)

    for ci, (opt, color) in enumerate(zip(OPTIMIZATIONS, colors)):
        cx = Inches(0.25) + ci * (col_w + col_gap)
        cy = Inches(1.45)
        add_rect(s, cx, cy, col_w, Inches(5.8), fill_color=BG_CARD)
        add_rect(s, cx, cy, col_w, Inches(0.05), fill_color=color)
        add_textbox(s, cx + Inches(0.1), cy + Inches(0.08),
                    col_w - Inches(0.2), Inches(0.38),
                    opt["category"], font_size=Pt(12), bold=True, color=color)
        bullet_list(s, opt["items"],
                    cx + Inches(0.08), cy + Inches(0.52),
                    col_w - Inches(0.18), Inches(5.0),
                    font_size=Pt(10), color=WHITE, bullet_color=color)


def slide_lessons_learned(prs):
    """Slide: key lessons from the optimization journey."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Lessons Learned",
                  subtitle="What actually moved the needle — and what didn't")

    lessons = [
        (ACCENT2, "✓ DID WORK",
         "Dispatch table (96→4 ops): single biggest win → +25% avg",
         "FP8→BF16 lossless cast: correct AND 4× less bandwidth",
         "BF16 tensor cores: 2× TFLOPS, no accuracy loss",
         "In-register route-weight fusion: free compute in epilogue",
         "Stable argsort: deterministic token ordering matters",
         "Bulk index_select: amortize gather across all experts",
        ),
        (RED_ERR, "✗ CAUSED REGRESSION",
         "cuBLAS TF32 without fused dequant: materialization cost > gain",
         "num_stages=4 in GEMM2: overflowed 228 KB SMEM, 3 errors",
         "Non-contiguous sorted_w tensor: Triton pointer crash",
         "Removing .contiguous() calls: Triton needs contiguous layout",
         "Removing routing_logits.to(float32): was ALREADY float32",
         "TF32 default on B200: caused ±4096 absolute errors",
        ),
        (ACCENT1, "→ KEY PRINCIPLES",
         "Verify SMEM budget: (BM×BK + BN×BK) × stages < 228 KB",
         "Read framework src FIRST: DPS, TorchBuilder, BuildSpec",
         "Profile by regime: small-T ≠ large-T bottlenecks",
         "Lossless casts > lossy precision for free accuracy",
         "Always .contiguous() before passing to Triton kernels",
         "Prefer epilogue fusion over separate kernel passes",
        ),
    ]

    card_w = (SLIDE_W - Inches(0.9)) / 3
    for ci, (color, hdr, *items) in enumerate(lessons):
        cx = Inches(0.3) + ci * (card_w + Inches(0.15))
        add_rect(s, cx, Inches(1.45), card_w, Inches(5.8), fill_color=BG_CARD)
        add_rect(s, cx, Inches(1.45), card_w, Inches(0.05), fill_color=color)
        add_textbox(s, cx + Inches(0.12), Inches(1.52), card_w - Inches(0.25),
                    Inches(0.38), hdr, font_size=Pt(13), bold=True, color=color)
        bullet_list(s, items, cx + Inches(0.12), Inches(1.98),
                    card_w - Inches(0.25), Inches(5.0),
                    font_size=Pt(11), color=WHITE, bullet_color=color)


def slide_future_work(prs):
    """Slide: future optimization directions."""
    s = blank_slide(prs)
    fill_bg(s, BG_DARK)
    add_title_bar(s, "Future Work & Next Steps",
                  subtitle="Where 5–10× additional speedup could come from")

    # High impact items with estimated gain boxes
    future = [
        ("CuTe DSL Grouped GEMM",
         "Batch all experts in ONE persistent kernel. Eliminates 32× sequential loop entirely.",
         "2–4×", ACCENT2),
        ("Native FP8 MMA Instructions",
         "Use Blackwell FP8 tensor core tiles via tl.dot on FP8 directly → ~4.5 PFLOPS.",
         "1.5–2×", ACCENT1),
        ("B200 L2 Cache Pinning",
         "126 MB L2 can hold expert weights for reuse. Pin frequently-used experts via"
         " cudaStreamAttrValue L2 policy.",
         "1.2–1.5×", ACCENT3),
        ("Triton Routing Kernel",
         "Replace 7 PyTorch ops (sigmoid, view, topk×3, scatter, gather) with 1 tl.program.",
         "1.2×", ACCENT1),
        ("Warp Specialization",
         "Blackwell feature: dedicate producer warps to HBM→SMEM loads, "
         "consumer warps to tensor core compute. Fully overlapped.",
         "1.3–1.8×", ACCENT2),
        ("Fused SwiGLU→GEMM2",
         "Eliminate c_buf write: stream SwiGLU output directly into GEMM2 K-loop. "
         "Saves Tk×2048×4 B per expert.",
         "1.1–1.3×", ACCENT3),
    ]

    cols = 3
    card_w = (SLIDE_W - Inches(0.9)) / cols
    card_h = Inches(2.35)
    gap = Inches(0.15)

    for i, (title, desc, gain, color) in enumerate(future):
        col = i % cols
        row = i // cols
        cx = Inches(0.3) + col * (card_w + gap)
        cy = Inches(1.45) + row * (card_h + gap)

        add_rect(s, cx, cy, card_w, card_h, fill_color=BG_CARD)
        add_rect(s, cx, cy, card_w, Inches(0.05), fill_color=color)
        # Gain badge
        add_rect(s, cx + card_w - Inches(0.85), cy + Inches(0.08),
                 Inches(0.75), Inches(0.32), fill_color=RGBColor(0x00, 0x30, 0x20))
        add_textbox(s, cx + card_w - Inches(0.85), cy + Inches(0.1),
                    Inches(0.75), Inches(0.28),
                    gain, font_size=Pt(10), bold=True, color=ACCENT2,
                    align=PP_ALIGN.CENTER)
        add_textbox(s, cx + Inches(0.1), cy + Inches(0.1),
                    card_w - Inches(1.1), Inches(0.38),
                    title, font_size=Pt(12), bold=True, color=color)
        add_textbox(s, cx + Inches(0.1), cy + Inches(0.52),
                    card_w - Inches(0.2), Inches(1.7),
                    desc, font_size=Pt(10.5), color=WHITE, wrap=True)


def build(prs):
    slide_speedup_journey(prs)
    slide_per_workload_table(prs)
    slide_small_vs_large(prs)
    slide_optimization_taxonomy(prs)
    slide_lessons_learned(prs)
    slide_future_work(prs)
    return prs


if __name__ == "__main__":
    prs = new_prs()
    build(prs)
    out = "moe_presentation_part3.pptx"
    prs.save(out)
    print(f"Saved: {out} ({prs.slides.__len__()} slides)")
