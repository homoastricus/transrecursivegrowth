from TRT import TRT
import math

print("=== TRT Demonstration ===")

trt_secure = TRT(mode='SECURE')

# Test different complexity levels
test_values = [1.0, 2.0, 3.0, 1.5, 2.5]  # Added more test values

for x in test_values:
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS FOR x = {x}")
    print(f"{'=' * 60}")

    # HCCSF Analysis
    hccsf_analysis = trt_secure.analyze_function(x, 'HCCSF')
    if 'error' not in hccsf_analysis:
        print(f"HCCSF value: {hccsf_analysis['result']:.6f}")
        print(f"Level: {hccsf_analysis['level_n']}.{hccsf_analysis['position_phi']:.3f}")
        print(f"log10(HCCSF): {hccsf_analysis['log10_result']:.3f}")
    else:
        print(f"HCCSF Error: {hccsf_analysis['error']}")

    print("-" * 40)

    # TRANSCEND Analysis
    transcend_analysis = trt_secure.analyze_function(x, 'TRANSCEND')
    if 'error' not in transcend_analysis:
        if transcend_analysis['is_overflow']:
            print("TRANSCEND: OVERFLOW (value too large)")
        elif transcend_analysis['is_infinite']:
            print("TRANSCEND: INFINITY")
        else:
            print(f"TRANSCEND: {transcend_analysis['result']:.2e}")
            print(f"log10(TRANSCEND): {transcend_analysis['log10_result']:.3f}")
        print(f"Computation time: {transcend_analysis['computation_time']:.4f} sec")
    else:
        print(f"TRANSCEND Error: {transcend_analysis['error']}")

    print("-" * 40)

    # Comparative analysis for higher functions
    if x <= 3.0:  # Limit to avoid overflow
        try:
            meta_analysis = trt_secure.analyze_function(x, 'META_TRANSCEND')
            if 'error' not in meta_analysis:
                if meta_analysis['is_overflow']:
                    print("META_TRANSCEND: OVERFLOW")
                else:
                    print(f"META_TRANSCEND: {meta_analysis['result']:.2e}")
                    print(f"log10(META): {meta_analysis['log10_result']:.3f}")
        except:
            print("META_TRANSCEND: computation impossible")

print(f"\n{'=' * 60}")
print("COMPARISON OF ALL FUNCTIONS FOR x = 1.5")
print(f"{'=' * 60}")

# Compare all functions for one value
comparison = trt_secure.compare_functions(1.5)
for func_name, analysis in comparison.items():
    if 'error' not in analysis:
        status = "∞" if analysis['is_infinite'] else "OVERFLOW" if analysis['is_overflow'] else "OK"
        if analysis['is_infinite'] or analysis['is_overflow']:
            value_str = status
        else:
            value_str = f"{analysis['result']:.2e}"

        print(
            f"{func_name:20} | {value_str:15} | log10: {analysis.get('log10_result', 'N/A'):8} | time: {analysis['computation_time']:.4f}s")

print(f"\n{'=' * 60}")
print("TESTING DIFFERENT MODES")
print(f"{'=' * 60}")

# Compare SECURE vs FULL for small values
test_small = [1.0, 2.0]
for x in test_small:
    print(f"\nx = {x}:")

    trt_secure = TRT(mode='SECURE')
    secure_result = trt_secure.TRANSCEND(x)

    trt_full = TRT(mode='FULL')
    full_result = trt_full.TRANSCEND(x)

    print(f"  SECURE: {secure_result:.2e}" if secure_result < 1e100 else "  SECURE: OVERFLOW")
    print(f"  FULL:   {full_result:.2e}" if full_result < 1e100 else "  FULL:   OVERFLOW")

print(f"\n{'=' * 60}")
print("ANALYZING BEHAVIOR AT LEVEL BOUNDARIES")
print(f"{'=' * 60}")

# Analyze behavior near integer boundaries
boundary_test = [0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1]
print("x      | HCCSF     | log10(TRANSCEND)")
print("-" * 40)
for x in boundary_test:
    hccsf = trt_secure.HCCSF_single_arg(x)
    transcend = trt_secure.TRANSCEND(x)
    log_transcend = math.log10(transcend) if transcend > 0 and transcend < 1e100 else float('inf')
    print(f"{x:6.3f} | {hccsf:8.4f} | {log_transcend:12.3f}")

print(f"\n{'=' * 60}")
print("TRT HIERARCHY GROWTH DEMONSTRATION")
print(f"{'=' * 60}")

# Demonstrate the growth hierarchy
print("\nGrowth progression from HCCSF to UNIVERSUM:")
demo_x = 1.0
functions_ordered = ['HCCSF', 'TRANSCEND', 'META_TRANSCEND', 'ULTIMATE_TRANSCEND', 'GOD_TRANSCEND', 'ABSOLUTE',
                     'UNIVERSUM']

for func_name in functions_ordered:
    try:
        analysis = trt_secure.analyze_function(demo_x, func_name)
        if 'error' not in analysis:
            if analysis['is_overflow'] or analysis['is_infinite']:
                status = "→ OVERFLOW"
            else:
                status = f"→ {analysis['result']:.2e}"
            print(f"{func_name:20} {status}")
        else:
            print(f"{func_name:20} → ERROR: {analysis['error']}")
    except:
        print(f"{func_name:20} → COMPUTATION FAILED")

print(f"\n{'=' * 60}")
print("GEOMETRIC COMPLEXITY ANALYSIS")
print(f"{'=' * 60}")

# Geometric complexity analysis
complexity_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
print("x     | Complexity Level | Approx. FGH Ordinal")
print("-" * 50)
for x in complexity_points:
    hccsf_val = trt_secure.HCCSF_single_arg(x)
    level = int(hccsf_val)

    # Map to approximate ordinal levels
    ordinal_map = {
        0: "ω",
        1: "ω^ω",
        2: "ε₀",
        3: "Γ₀",
        4: "ψ(Ω^Ω^ω)",
        5: "ψ(Ω_ω)",
        6: "ψ(ε_{Ω+1})",
        7: "ψ(Ω_Ω)",
        8: "ψ(I)",
        9: "ψ(M)"
    }

    ordinal = ordinal_map.get(level, "Transcendental")
    print(f"{x:5.1f} | {level:15d} | {ordinal:>20}")

print(f"\n{'=' * 60}")
print("DEMONSTRATION COMPLETED")
print(f"{'=' * 60}")