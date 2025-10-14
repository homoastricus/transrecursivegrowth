import math

from TRT import TRT

print("=== TRT демонстрация ===")

trt_secure = TRT(mode='SECURE')

# Тестируем разные уровни сложности
test_values = [1.0, 2.0, 3.0, 1.5, 2.5]  # Добавил больше тестовых значений

for x in test_values:
    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ ДЛЯ x = {x}")
    print(f"{'=' * 60}")

    # Анализ HCCSF
    hccsf_analysis = trt_secure.analyze_function(x, 'HCCSF')
    if 'error' not in hccsf_analysis:
        print(f"HCCSF значение: {hccsf_analysis['result']:.6f}")
        print(f"Уровень: {hccsf_analysis['level_n']}.{hccsf_analysis['position_phi']:.3f}")
        print(f"log10(HCCSF): {hccsf_analysis['log10_result']:.3f}")
    else:
        print(f"Ошибка HCCSF: {hccsf_analysis['error']}")

    print("-" * 40)

    # Анализ TRANSCEND
    transcend_analysis = trt_secure.analyze_function(x, 'TRANSCEND')
    if 'error' not in transcend_analysis:
        if transcend_analysis['is_overflow']:
            print("TRANSCEND: ПЕРЕПОЛНЕНИЕ (значение слишком велико)")
        elif transcend_analysis['is_infinite']:
            print("TRANSCEND: БЕСКОНЕЧНОСТЬ")
        else:
            print(f"TRANSCEND: {transcend_analysis['result']:.2e}")
            print(f"log10(TRANSCEND): {transcend_analysis['log10_result']:.3f}")
        print(f"Время вычисления: {transcend_analysis['computation_time']:.4f} сек")
    else:
        print(f"Ошибка TRANSCEND: {transcend_analysis['error']}")

    print("-" * 40)

    # Сравнительный анализ для более высоких функций
    if x <= 3.0:  # Ограничиваем для избежания переполнения
        try:
            meta_analysis = trt_secure.analyze_function(x, 'META_TRANSCEND')
            if 'error' not in meta_analysis:
                if meta_analysis['is_overflow']:
                    print("META_TRANSCEND: ПЕРЕПОЛНЕНИЕ")
                else:
                    print(f"META_TRANSCEND: {meta_analysis['result']:.2e}")
                    print(f"log10(META): {meta_analysis['log10_result']:.3f}")
        except:
            print("META_TRANSCEND: вычисление невозможно")

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ ВСЕХ ФУНКЦИЙ ДЛЯ x = 1.5")
print(f"{'=' * 60}")

# Сравнение всех функций для одного значения
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
print("ТЕСТИРОВАНИЕ РАЗНЫХ РЕЖИМОВ")
print(f"{'=' * 60}")

# Сравнение SECURE vs FULL для малых значений
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
print("АНАЛИЗ ПОВЕДЕНИЯ НА ГРАНИЦАХ УРОВНЕЙ")
print(f"{'=' * 60}")

# Анализ поведения около целых чисел
boundary_test = [0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1]
print("x      | HCCSF     | log10(TRANSCEND)")
print("-" * 40)
for x in boundary_test:
    hccsf = trt_secure.HCCSF_single_arg(x)
    transcend = trt_secure.TRANSCEND(x)
    log_transcend = math.log10(transcend) if transcend > 0 and transcend < 1e100 else float('inf')
    print(f"{x:6.3f} | {hccsf:8.4f} | {log_transcend:12.3f}")

print(f"\n{'=' * 60}")
print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
print(f"{'=' * 60}")