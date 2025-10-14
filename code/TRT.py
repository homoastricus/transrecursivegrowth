import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Callable, Literal, Tuple, Optional, Any
from decimal import Decimal, getcontext
from functools import lru_cache
import warnings
import time


class TRT:
    """
    Полная реализация Трансрекурсивной Теории Роста
    Режимы:
    - 'FULL': строгая математическая реализация
    - 'SECURE': безопасный режим с ограничениями
    """

    def __init__(self, mode: Literal['FULL', 'SECURE'] = 'SECURE', precision: int = 1000):
        self.mode = mode
        self.precision = precision
        getcontext().prec = precision

        if self.mode == 'FULL':
            self._init_full_mode()
        else:
            self._init_secure_mode()

        self._clear_caches()

        # Уровни сложности
        self.complexity_levels = {
            0: "Линейный/полиномиальный рост",
            1: "Экспоненциальный рост",
            2: "Двойная экспонента",
            3: "Тройная экспонента",
            4: "Тетрация",
            5: "Пентация",
            6: "Гипероператоры высших порядков",
            7: "Трансфинитная рекурсия",
            8: "Мета-рекурсивный рост",
            9: "Онтологический рост"
        }

    def _clear_caches(self):
        """Очистка кэшей между вычислениями"""
        self._hccsf_cache = {}
        self._t_n_cache = {}
        self._transcend_cache = {}
        self._meta_transcend_cache = {}

    def _init_full_mode(self):
        """Режим FULL - максимальная математическая точность"""
        self.max_depth = 100
        self.max_level = 100
        self.overflow_threshold = 1e308
        self.max_iterations = 10000
        self.max_meta_iterations = 100000
        self.max_ultimate_depth = 100
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    def _init_secure_mode(self):
        """Режим SECURE - безопасные ограничения"""
        self.max_depth = 20
        self.max_level = 30
        self.overflow_threshold = 1e100
        self.max_iterations = 1000
        self.max_meta_iterations = 5000
        self.max_ultimate_depth = 10

    def S(self, t: float) -> float:
        """Сглаживающая функция S(t) = t/(1+t)"""
        if t > self.overflow_threshold:
            return 1.0
        if t < 0:
            return 0.0
        try:
            return t / (1 + t)
        except (OverflowError, ZeroDivisionError):
            return 1.0

    def T0(self, phi: float) -> float:
        """Базовый генератор уровня 0: T₀(φ) = 1/(1-φ)"""
        if phi >= 1.0:
            return self.overflow_threshold
        if phi <= 0.0:
            return 1.0

        if self.mode == 'SECURE' and phi >= 0.999:
            return self.overflow_threshold

        try:
            return 1.0 / (1.0 - phi)
        except ZeroDivisionError:
            return self.overflow_threshold

    def Tn(self, phi: float, n: int) -> float:
        """Рекурсивный генератор уровня n: Tₙ(φ) = exp(Tₙ₋₁(φ))"""
        if n < 0:
            return max(0.0, phi)  # Линейная экстраполяция для n < 0

        if n == 0:
            return self.T0(phi)

        if self.mode == 'SECURE' and n > self.max_depth:
            return self.overflow_threshold

        cache_key = (phi, n)
        if cache_key in self._t_n_cache:
            return self._t_n_cache[cache_key]

        try:
            prev = self.Tn(phi, n - 1)

            if prev >= self.overflow_threshold:
                result = self.overflow_threshold
            elif prev > 700:  # exp(700) уже слишком большое
                result = self.overflow_threshold
            else:
                result = math.exp(prev)

            self._t_n_cache[cache_key] = result
            return result

        except (OverflowError, ValueError):
            return self.overflow_threshold

    def HCCSF(self, n: int, phi: float) -> float:
        """Функция масштабирования иерархической вычислимой сложности"""
        if n < 0:
            return max(0.0, phi)

        cache_key = (n, phi)
        if cache_key in self._hccsf_cache:
            return self._hccsf_cache[cache_key]

        if self.mode == 'SECURE' and n > self.max_level:
            n = self.max_level
            phi = min(phi, 0.999)

        if n == 0:
            result = self.S(self.T0(phi))
        else:
            try:
                Tn_phi = self.Tn(phi, n)
                Tn_0 = self.Tn(0.0, n)

                S_Tn_phi = self.S(Tn_phi)
                S_Tn_0 = self.S(Tn_0)

                numerator = S_Tn_phi - S_Tn_0
                denominator = 1 - S_Tn_0

                if abs(denominator) < 1e-300:
                    result = n + 0.999999999
                else:
                    fractional = numerator / denominator
                    result = n + max(0.0, min(0.999999999, fractional))

            except (OverflowError, ZeroDivisionError):
                result = n + 0.999999999

        self._hccsf_cache[cache_key] = result
        return result

    def HCCSF_single_arg(self, x: float) -> float:
        """Удобная обертка: HCCSF(x) где x = n + φ"""
        n = int(x)
        phi = x - n
        return self.HCCSF(n, phi)

    def META_ITER(self, G: float, k: int) -> float:
        """Оператор мета-итерации"""
        if k <= 0:
            return G

        current = G
        max_k = min(k, self.max_meta_iterations) if self.mode == 'SECURE' else k

        for i in range(max_k):
            try:
                if current >= self.overflow_threshold:
                    return self.overflow_threshold

                inner_arg = 1 - math.exp(-current)
                n_level = max(0, int(math.floor(G)))

                hccsf_val = self.HCCSF(n_level, inner_arg)

                if hccsf_val >= 700:  # exp(700) уже слишком большое
                    return self.overflow_threshold

                current = math.exp(hccsf_val)

            except (OverflowError, ValueError):
                return self.overflow_threshold

        return current

    def TRANSCEND(self, x: float) -> float:
        """Трансцендентная функция роста"""
        self._clear_caches()

        n = int(x)
        phi = x - n

        # Вычисляем T_{n-1}(φ)
        if n == 0:
            T_prev = self.T0(phi)
        else:
            T_prev = self.Tn(phi, n - 1)

        if T_prev >= self.overflow_threshold:
            return float('inf')

        # Вычисляем G
        try:
            exp_arg = -T_prev
            if exp_arg < -700:  # exp(-700) практически 0
                inner = 1.0
            else:
                inner = 1 - math.exp(exp_arg)

            G = self.HCCSF(n, inner)
        except (OverflowError, ValueError):
            return float('inf')

        # Вычисляем k
        try:
            if G >= 700:
                k = self.max_iterations
            else:
                k = int(math.exp(G))
                k = min(k, self.max_iterations) if self.mode == 'SECURE' else k
        except (OverflowError, ValueError):
            k = self.max_iterations

        # META_ITER
        meta_result = self.META_ITER(G, k)

        if meta_result >= self.overflow_threshold:
            return float('inf')

        # Финальное вычисление
        try:
            power_base = meta_result
            power_exp = T_prev

            # Защита от переполнения
            if power_base <= 0:
                power_result = 0
            elif power_exp * math.log(max(1, power_base)) > 700:
                return float('inf')
            else:
                power_result = power_base ** power_exp

            if power_result > 700:
                return float('inf')

            result = math.exp(power_result)
            return min(result, self.overflow_threshold)

        except (OverflowError, ValueError):
            return float('inf')

    def META_TRANSCEND(self, x: float) -> float:
        """Мета-трансцендентная функция роста"""
        self._clear_caches()

        try:
            limiter = self.TRANSCEND(x)
            if limiter >= self.overflow_threshold:
                return float('inf')

            limiter_int = min(int(limiter), self.max_iterations)

            result = 1.0
            for k in range(1, limiter_int + 1):
                if result >= self.overflow_threshold:
                    return float('inf')
                transcend_val = self.TRANSCEND(k + x)
                result *= transcend_val
                if result >= self.overflow_threshold:
                    return float('inf')

            return result

        except (OverflowError, ValueError):
            return float('inf')

    def ULTIMATE_TRANSCEND(self, x: float) -> float:
        """Ультимативная трансцендентная функция"""
        self._clear_caches()

        try:
            UPPER = self.META_TRANSCEND(x)
            if UPPER >= self.overflow_threshold:
                return float('inf')

            upper_int = min(int(UPPER), self.max_ultimate_depth)

            def nested_loop(depth: int, current: float) -> float:
                if depth <= 0:
                    return self.META_TRANSCEND(current)

                result = current
                iterations = min(int(UPPER), self.max_iterations)

                for i in range(iterations):
                    if result >= self.overflow_threshold:
                        return float('inf')
                    result = nested_loop(depth - 1, result)

                return result

            return nested_loop(upper_int, x)

        except (OverflowError, ValueError, RecursionError):
            return float('inf')

    def GOD_TRANSCEND(self, x: float) -> float:
        """Божественная трансцендентная функция"""
        self._clear_caches()

        try:
            OMEGA = self.ULTIMATE_TRANSCEND(x)
            if OMEGA >= self.overflow_threshold:
                return float('inf')

            omega_int = min(int(OMEGA), self.max_ultimate_depth)

            def hyper_nested_loop(current_omega: float, depth: int, current_value: float) -> float:
                if current_omega <= 0 or depth > self.max_depth:
                    return current_value

                result = current_value
                iterations = max(1, min(int(current_omega) - 1, self.max_iterations))

                for i in range(iterations):
                    if result >= self.overflow_threshold:
                        return float('inf')
                    result = self.ULTIMATE_TRANSCEND(result)

                return hyper_nested_loop(current_omega - 1, depth + 1, result)

            result = x
            main_iterations = min(omega_int, self.max_iterations)

            for i in range(main_iterations):
                if result >= self.overflow_threshold:
                    return float('inf')
                result = hyper_nested_loop(OMEGA, 0, result)

            return result

        except (OverflowError, ValueError, RecursionError):
            return float('inf')

    def ABSOLUTE(self, x: float) -> float:
        """Абсолютная трансцендентная функция с фрактальной структурой"""
        self._clear_caches()

        try:
            # Базовая величина через GOD_TRANSCEND
            V0 = self.GOD_TRANSCEND(x)
            if V0 >= self.overflow_threshold:
                return float('inf')

            # Функция ветвления B(V)
            def B(V: float) -> float:
                return math.log(max(1, V)) + 1

            # Функция дочерних узлов U(V)
            def U(V: float) -> float:
                return self.GOD_TRANSCEND(V)

            # Агрегатор
            def Agg(values: List[float]) -> float:
                return max(values) if values else 1.0

            # Рекурсивная функция дерева
            def build_tree(current_value: float, max_depth: int = 5) -> float:
                if max_depth <= 0:
                    return current_value

                b = B(current_value)
                k = int(b)
                delta = b - k

                children = []
                for i in range(k + 2):  # k и k+1 узлов
                    if len(children) >= self.max_iterations:
                        break
                    child_val = U(current_value + i * 0.1)
                    children.append(child_val)

                if not children:
                    return current_value

                Rk = Agg(children[:k]) if k <= len(children) else current_value
                Rk1 = Agg(children[:k + 1]) if k + 1 <= len(children) else Rk

                # Логарифмическая интерполяция
                if Rk <= 0 or Rk1 <= 0:
                    R_interp = max(Rk, Rk1)
                else:
                    log_Rk = math.log(Rk)
                    log_Rk1 = math.log(Rk1)
                    log_R = (1 - delta) * log_Rk + delta * log_Rk1
                    R_interp = math.exp(log_R)

                return build_tree(R_interp, max_depth - 1)

            max_tree_depth = min(5, self.max_depth)
            result = build_tree(V0, max_tree_depth)
            return min(result, self.overflow_threshold)

        except (OverflowError, ValueError, RecursionError):
            return float('inf')

    def UNIVERSUM(self, x: float) -> float:
        """Функция UNIVERSUM - предел вычислимого роста"""
        self._clear_caches()

        try:
            # Базовое значение через ABSOLUTE
            E = self.ABSOLUTE(x)
            if E >= self.overflow_threshold:
                return float('inf')

            # Глубина композиции
            L = math.exp(E)
            L_int = min(int(L), self.max_iterations)

            # Набор функций TRT
            functions = [
                self.TRANSCEND,
                self.META_TRANSCEND,
                self.ULTIMATE_TRANSCEND,
                self.GOD_TRANSCEND,
                self.ABSOLUTE
            ]

            # Рекурсивная генерация композиций
            def generate_compositions(funcs: List[Callable], max_depth: int) -> List[Callable]:
                if max_depth <= 0:
                    return [lambda x: x]  # Тождественная функция

                compositions = []
                prev_level = generate_compositions(funcs, max_depth - 1)

                for prev_comp in prev_level:
                    for f in funcs:
                        def composed(x, f1=prev_comp, f2=f):
                            return f2(f1(x))

                        compositions.append(composed)

                return prev_level + compositions

            # Генерация композиций
            comp_depth = min(3, self.max_depth)  # Ограничиваем глубину для практичности
            all_compositions = generate_compositions(functions, comp_depth)

            # Вычисление максимума
            best = E
            for comp in all_compositions[:min(100, len(all_compositions))]:  # Ограничиваем количество
                try:
                    val = comp(E)
                    if val > best and val < self.overflow_threshold:
                        best = val
                except:
                    continue

            return best

        except (OverflowError, ValueError, RecursionError):
            return float('inf')

    # Аналитические методы
    def analyze_function(self, x: float, func_name: str) -> Dict[str, Any]:
        """Анализ функции в точке x"""
        func_map = {
            'HCCSF': self.HCCSF_single_arg,
            'TRANSCEND': self.TRANSCEND,
            'META_TRANSCEND': self.META_TRANSCEND,
            'ULTIMATE_TRANSCEND': self.ULTIMATE_TRANSCEND,
            'GOD_TRANSCEND': self.GOD_TRANSCEND,
            'ABSOLUTE': self.ABSOLUTE,
            'UNIVERSUM': self.UNIVERSUM
        }

        if func_name not in func_map:
            return {'error': f'Функция {func_name} не найдена'}

        try:
            start_time = time.time()
            result = func_map[func_name](x)
            compute_time = time.time() - start_time

            n = int(x)
            phi = x - n

            analysis = {
                'input': x,
                'function': func_name,
                'result': result,
                'computation_time': compute_time,
                'level_n': n,
                'position_phi': phi,
                'log10_result': math.log10(result) if result > 0 else -float('inf'),
                'mode': self.mode,
                'is_overflow': result >= self.overflow_threshold,
                'is_infinite': math.isinf(result)
            }

            return analysis

        except Exception as e:
            return {'error': str(e), 'input': x, 'function': func_name}

    def compare_functions(self, x: float) -> Dict[str, Any]:
        """Сравнение всех TRT функций в точке x"""
        functions = ['HCCSF', 'TRANSCEND', 'META_TRANSCEND', 'ULTIMATE_TRANSCEND', 'GOD_TRANSCEND', 'ABSOLUTE',
                     'UNIVERSUM']

        results = {}
        for func in functions:
            analysis = self.analyze_function(x, func)
            results[func] = analysis

        return results


# Визуализация
class TRTVisualizer:
    """Класс для визуализации TRT функций"""

    def __init__(self, trt: TRT):
        self.trt = trt

    def plot_function(self, func_name: str, start: float, end: float, num_points: int = 1000):
        """Построение графика TRT функции"""
        x_values = np.linspace(start, end, num_points)
        y_values = []
        log_y_values = []

        print(f"Построение {func_name} на [{start}, {end}]...")

        for i, x in enumerate(x_values):
            if i % 100 == 0:
                print(f"Прогресс: {i}/{num_points}")

            try:
                analysis = self.trt.analyze_function(x, func_name)
                if 'result' in analysis:
                    val = analysis['result']
                    y_values.append(val)
                    if val > 0 and not math.isinf(val):
                        log_y_values.append(math.log10(val))
                    else:
                        log_y_values.append(np.nan)
                else:
                    y_values.append(np.nan)
                    log_y_values.append(np.nan)
            except:
                y_values.append(np.nan)
                log_y_values.append(np.nan)

        # Построение графиков
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # График 1: Функция
        ax1.plot(x_values, y_values, 'b-', linewidth=2, label=func_name)
        ax1.set_xlabel('x')
        ax1.set_ylabel(func_name + '(x)')
        ax1.set_title(f'Функция {func_name}(x) на интервале [{start}, {end}]')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')

        # График 2: Логарифм
        ax2.plot(x_values, log_y_values, 'r-', linewidth=2, label='log₁₀(' + func_name + '(x))')
        ax2.set_xlabel('x')
        ax2.set_ylabel('log₁₀(' + func_name + '(x))')
        ax2.set_title(f'Логарифм {func_name}(x) на интервале [{start}, {end}]')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        return x_values, y_values, log_y_values


# Пример использования
if __name__ == "__main__":
    # Создание экземпляра TRT
    trt = TRT(mode='SECURE', precision=1000)

    # Анализ функций
    print("=== Анализ TRANSCEND ===")
    analysis = trt.analyze_function(2.0, 'TRANSCEND')
    for key, value in analysis.items():
        print(f"{key}: {value}")

    print("\n=== Сравнение функций при x=1.5 ===")
    comparison = trt.compare_functions(1.5)
    for func, result in comparison.items():
        if 'result' in result:
            print(f"{func}: {result['result']:.2e} (log10: {result.get('log10_result', 'N/A'):.2f})")

    # Визуализация
    visualizer = TRTVisualizer(trt)
    visualizer.plot_function('TRANSCEND', 0, 2, 500)