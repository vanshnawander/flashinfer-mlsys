"""
build_presentation.py — combines all parts into one .pptx
Run from the presentation/ directory:
    cd presentation && python build_presentation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from style import new_prs
import slides_1_title_problem as p1
import slides_2_solutions as p2
import slides_3_results as p3

def main():
    prs = new_prs()
    print("Building Part 1: Title + Problem...")
    p1.build(prs)
    print("Building Part 2: Solution Approaches...")
    p2.build(prs)
    print("Building Part 3: Results + Future Work...")
    p3.build(prs)

    out = os.path.join(os.path.dirname(__file__), "moe_b200_optimization.pptx")
    prs.save(out)
    n = len(prs.slides)
    print(f"\n✓ Saved: {out}")
    print(f"  Total slides: {n}")
    print(f"  Slides 1–5:   Title, Problem, Hardware, Pipeline, Workloads")
    print(f"  Slides 6–10:  Sub-1 Baseline, Sub-3 Dispatch, Sub-9 BF16-TC, Sub-13 Epilogue, CUDA path")
    print(f"  Slides 11–16: Speedup Journey, Per-workload Table, Bottlenecks, Taxonomy, Lessons, Future")
    return out

if __name__ == "__main__":
    main()
