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
    Удобства:
    - cache_limit: максимальное число записей в каждом кэше (умное очищение)
    - overflow_softness: параметр "мягкости" при переполнении (меньше - мягче)
    """

    def __init__(self,
                 mode: Literal['FULL', 'SECURE'] = 'SECURE',
                 precision: int = 1000,
                 cache_limit: int = 100000,
                 overflow_softness: float = 1.0):
        self.mode = mode
        self.precision = precision
        getcontext().prec = precision

        # Кэш и лимиты
        self.cache_limit = cache_limit
        self.overflow_softness = max(1e-9, float(overflow_softness))  # >0

        # Инициализация режимов
        if self.mode == 'FULL':
            self._init_full_mode()
        else:
            self._init_secure_mode()

        # кэши и первичная очистка (вызывается один раз)
        self._clear_caches()

        # Уровни сложности (информативно)
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
        """Очистка кэшей между сессиями (вызовется в init или вручную)"""
        self._hccsf_cache = {}
        self._t_n_cache = {}
        self._transcend_cache = {}
        self._meta_transcend_cache = {}

    def clear_all(self):
        """Публичный метод для ручной очистки всех кэшей"""
        self._clear_caches()

    def _manage_cache(self, cache: dict):
        """Ограничиваем размер кэша: при переполнении удаляем старейшие записи (приближенно)."""
        if not isinstance(cache, dict):
            return
        if len(cache) > self.cache_limit:
            # удаляем примерно половину старейших элементов
            remove_count = max(1, len(cache) // 2)
            # Итерация по dict в Python 3.7+ детерминирована (insertion order)
            for _ in range(remove_count):
                try:
                    cache.pop(next(iter(cache)))
                except StopIteration:
                    break

    def _init_full_mode(self):
        """Режим FULL - максимальная математическая точность"""
        self.max_depth = 100
        self.max_level = 100
        self.overflow_threshold = 1e308
        self.max_iterations = 1000000
        self.max_meta_iterations = 1000000
        self.max_ultimate_depth = 1000
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    def _init_secure_mode(self):
        """Режим SECURE - безопасные ограничения"""
        self.max_depth = 20
        self.max_level = 30
        self.overflow_threshold = 1e100
        self.max_iterations = 1000
        self.max_meta_iterations = 5000
        self.max_ultimate_depth = 50

    # --------------------------
    # Soft clip (мягкое обрезание)
    # --------------------------
    def soft_clip(self, x: float) -> float:
        """
        Мягкое ограничение: при значениях порядка overflow_threshold и выше возвращаем
        возрастающую, но плавную аппроксимацию, зависящую от overflow_softness.
        При overflow_softness -> large поведение приближается к "жесткому" порогу.
        При overflow_softness -> small сглаживание сильнее.
        """
        try:
            th = float(self.overflow_threshold)
            if x <= th:
                return x
            # Пусть рост после порога идёт логарифмически, масштабируемо по softness.
            # Формула: th * (1 + softness * log1p((x-th)/th))
            # Это монотонно растёт, но медленнее, чем линейно, не даёт inf.
            frac = (x - th) / th
            # избегаем отрицательных/чрезмерных аргументов
            val = th * (1.0 + self.overflow_softness * math.log1p(max(0.0, frac)))
            # защита от NaN/inf:
            if math.isnan(val) or math.isinf(val):
                return th * (1.0 + self.overflow_softness)
            return val
        except Exception:
            return float(self.overflow_threshold)

    # --------------------------
    # Базовые операторы
    # --------------------------
    def S(self, t: float) -> float:
        """Сглаживающая функция S(t) = t/(1+t)"""
        if t <= 0:
            return 0.0
        # если t слишком велико, возвращаем значение, близкое к 1.0
        if t >= self.overflow_threshold:
            return 1.0
        # стандартная формула
        return t / (1.0 + t)

    def T0(self, phi: float) -> float:
        """Базовый генератор уровня 0: T₀(φ) = 1/(1-φ)"""
        if phi >= 1.0:
            return self.soft_clip(self.overflow_threshold)
        if phi <= 0.0:
            return 1.0
        if self.mode == 'SECURE' and phi >= 0.999999:
            return self.soft_clip(self.overflow_threshold)
        try:
            return 1.0 / (1.0 - phi)
        except ZeroDivisionError:
            return self.soft_clip(self.overflow_threshold)

    def Tn(self, phi: float, n: int) -> float:
        """Рекурсивный генератор уровня n: Tₙ(φ) = exp(Tₙ₋₁(φ))"""
        if n < 0:
            return max(0.0, phi)

        if n == 0:
            return self.T0(phi)

        if self.mode == 'SECURE' and n > self.max_depth:
            return self.soft_clip(self.overflow_threshold)

        cache_key = (phi, n)
        if cache_key in self._t_n_cache:
            return self._t_n_cache[cache_key]

        try:
            prev = self.Tn(phi, n - 1)
            if prev >= self.overflow_threshold:
                result = self.soft_clip(prev)
            else:
                # позволяем строгую экспоненту; если переполнение, применяем soft_clip
                try:
                    result = math.exp(prev)
                except OverflowError:
                    result = self.soft_clip(prev * 10.0)
            # записываем в кэш и управляем размером
            self._t_n_cache[cache_key] = result
            self._manage_cache(self._t_n_cache)
            return result
        except Exception:
            return self.soft_clip(self.overflow_threshold)

    def HCCSF(self, n: int, phi: float) -> float:
        """Функция масштабирования иерархической вычислимой сложности"""
        if n < 0:
            return max(0.0, phi)

        cache_key = (n, phi)
        if cache_key in self._hccsf_cache:
            return self._hccsf_cache[cache_key]

        if self.mode == 'SECURE' and n > self.max_level:
            n = self.max_level
            phi = min(phi, 0.999999)

        try:
            if n == 0:
                result = self.S(self.T0(phi))
            else:
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
        except Exception:
            result = n + 0.999999999

        self._hccsf_cache[cache_key] = result
        self._manage_cache(self._hccsf_cache)
        return result

    def HCCSF_single_arg(self, x: float) -> float:
        """Удобная обертка: HCCSF(x) где x = n + φ"""
        n = int(math.floor(x))
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
                if current <= 0:
                    inner_arg = 1.0 - math.exp(-max(1e-16, current))
                else:
                    inner_arg = 1.0 - math.exp(-current)

                n_level = max(0, int(math.floor(G)))
                hccsf_val = self.HCCSF(n_level, inner_arg)

                # допускаем большие значения и применяем мягкое обрезание при необходимости
                if hccsf_val >= self.overflow_threshold:
                    return self.soft_clip(hccsf_val)

                # экспоненцируем строго; если переполнение — возвращаем soft_clip
                try:
                    current = math.exp(hccsf_val)
                except OverflowError:
                    return self.soft_clip(hccsf_val * 10.0)

            except Exception:
                return self.soft_clip(self.overflow_threshold)

        return current

    def TRANSCEND(self, x: float) -> float:
        """Трансцендентная функция роста"""
        # НЕ очищаем кэши тут — пользователь контролирует это через clear_all()

        n = int(math.floor(x))
        phi = x - n

        # Вычисляем T_{n-1}(φ)
        if n == 0:
            T_prev = self.T0(phi)
        else:
            T_prev = self.Tn(phi, n - 1)

        if T_prev >= self.overflow_threshold:
            return self.soft_clip(T_prev)

        # Вычисляем G
        try:
            exp_arg = -T_prev
            if exp_arg < -700:
                inner = 1.0
            else:
                inner = 1 - math.exp(exp_arg)
            G = self.HCCSF(n, inner)
        except Exception:
            return self.soft_clip(self.overflow_threshold)

        # Вычисляем k
        try:
            if G >= 700:
                k = min(self.max_iterations, int(1e6))
            else:
                k = int(math.exp(G))
                if self.mode == 'SECURE':
                    k = min(k, self.max_iterations)
        except Exception:
            k = self.max_iterations

        # META_ITER
        meta_result = self.META_ITER(G, k)

        if meta_result >= self.overflow_threshold:
            return self.soft_clip(meta_result)

        # Финальное вычисление: power_base ** power_exp then exp(...)
        try:
            power_base = meta_result
            power_exp = T_prev

            if power_base <= 0:
                power_result = 0.0
            else:
                # try pow with protection
                try:
                    # Если log слишком велик, применяем soft_clip
                    if power_exp * math.log(max(1.0, power_base)) > 1e6:
                        # экстренное смягчение
                        return self.soft_clip(power_base ** min(power_exp, 1e6))
                    power_result = math.pow(power_base, power_exp)
                except OverflowError:
                    return self.soft_clip(self.overflow_threshold)

            # теперь exp(power_result) — строго, но смягчаем при переполнении
            try:
                final = math.exp(power_result)
            except OverflowError:
                return self.soft_clip(power_result)

            return min(final, self.soft_clip(final))
        except Exception:
            return self.soft_clip(self.overflow_threshold)

    def META_TRANSCEND(self, x: float) -> float:
        """Мета-трансцендентная функция роста"""
        # НЕ очищаем кэши тут
        try:
            limiter = self.TRANSCEND(x)
            if limiter >= self.overflow_threshold:
                return self.soft_clip(limiter)

            limiter_int = min(int(max(0, limiter)), self.max_iterations)
            result = 1.0
            for k in range(1, limiter_int + 1):
                transcend_val = self.TRANSCEND(k + x)
                result *= transcend_val
                if result >= self.overflow_threshold:
                    return self.soft_clip(result)
            return result
        except Exception:
            return self.soft_clip(self.overflow_threshold)

    def ULTIMATE_TRANSCEND(self, x: float) -> float:
        """Ультимативная трансцендентная функция"""
        # НЕ очищаем кэши тут
        try:
            UPPER = self.META_TRANSCEND(x)
            if UPPER >= self.overflow_threshold:
                return self.soft_clip(UPPER)

            upper_int = min(int(max(0, UPPER)), self.max_ultimate_depth)

            # рекурсивная вложенность; осторожно с глубиной
            def nested_loop(depth: int, current: float) -> float:
                if depth <= 0:
                    return self.META_TRANSCEND(current)
                result = current
                iterations = min(int(max(1, UPPER)), self.max_iterations)
                for i in range(iterations):
                    result = nested_loop(depth - 1, result)
                    if result >= self.overflow_threshold:
                        return self.soft_clip(result)
                return result

            return nested_loop(upper_int, x)
        except Exception:
            return self.soft_clip(self.overflow_threshold)

    def GOD_TRANSCEND(self, x: float) -> float:
        """Божественная трансцендентная функция"""
        # НЕ очищаем кэши тут
        try:
            OMEGA = self.ULTIMATE_TRANSCEND(x)
            if OMEGA >= self.overflow_threshold:
                return self.soft_clip(OMEGA)

            omega_int = min(int(max(0, OMEGA)), self.max_ultimate_depth)

            def hyper_nested_loop(current_omega: float, depth: int, current_value: float) -> float:
                if current_omega <= 0 or depth > self.max_depth:
                    return current_value
                result = current_value
                iterations = max(1, min(int(max(0, math.floor(current_omega))) - 1, self.max_iterations))
                for i in range(iterations):
                    result = self.ULTIMATE_TRANSCEND(result)
                    if result >= self.overflow_threshold:
                        return self.soft_clip(result)
                return hyper_nested_loop(current_omega - 1, depth + 1, result)

            result = x
            main_iterations = min(omega_int, self.max_iterations)
            for i in range(main_iterations):
                result = hyper_nested_loop(OMEGA, 0, result)
                if result >= self.overflow_threshold:
                    return self.soft_clip(result)
            return result
        except Exception:
            return self.soft_clip(self.overflow_threshold)

    def ABSOLUTE(self, x: float) -> float:
        """Абсолютная трансцендентная функция с фрактальной структурой"""
        # НЕ очищаем кэши тут
        try:
            V0 = self.GOD_TRANSCEND(x)
            if V0 >= self.overflow_threshold:
                return self.soft_clip(V0)

            def B(V: float) -> float:
                return math.log(max(1.0, V)) + 1.0

            def U(V: float) -> float:
                return self.GOD_TRANSCEND(V)

            def Agg(values: List[float]) -> float:
                return max(values) if values else 1.0

            def build_tree(current_value: float, max_depth: int = 5) -> float:
                if max_depth <= 0:
                    return current_value
                b = B(current_value)
                k = int(math.floor(max(0.0, b)))
                delta = b - k
                children = []
                # ограничиваем число детей практично
                child_limit = min(self.max_iterations, k + 2)
                for i in range(child_limit):
                    child_val = U(current_value + i * 0.1)
                    children.append(child_val)
                    if children[-1] >= self.overflow_threshold:
                        children[-1] = self.soft_clip(children[-1])
                if not children:
                    return current_value
                Rk = Agg(children[:k]) if k > 0 and k <= len(children) else children[0]
                Rk1 = Agg(children[:k+1]) if (k + 1) <= len(children) else Rk
                if Rk <= 0 or Rk1 <= 0:
                    R_interp = max(Rk, Rk1)
                else:
                    log_Rk = math.log(Rk)
                    log_Rk1 = math.log(Rk1)
                    log_R = (1.0 - delta) * log_Rk + delta * log_Rk1
                    R_interp = math.exp(log_R)
                return build_tree(R_interp, max_depth - 1)

            max_tree_depth = min(5, self.max_depth)
            result = build_tree(V0, max_tree_depth)
            if result >= self.overflow_threshold:
                return self.soft_clip(result)
            return result
        except Exception:
            return self.soft_clip(self.overflow_threshold)

    def UNIVERSUM(self, x: float) -> float:
        """Функция UNIVERSUM - предел вычислимого роста"""
        # НЕ очищаем кэши тут
        try:
            E = self.ABSOLUTE(x)
            if E >= self.overflow_threshold:
                return self.soft_clip(E)

            # Глубина композиции (в ограниченном виде для практичности)
            try:
                L = math.exp(min(E, 700))  # ограничиваем exp для практичности
            except Exception:
                L = self.overflow_threshold
            L_int = min(int(max(0, L)), self.max_iterations)

            functions = [
                self.TRANSCEND,
                self.META_TRANSCEND,
                self.ULTIMATE_TRANSCEND,
                self.GOD_TRANSCEND,
                self.ABSOLUTE
            ]

            def generate_compositions(funcs: List[Callable], max_depth: int) -> List[Callable]:
                if max_depth <= 0:
                    return [lambda x: x]
                compositions = []
                prev_level = generate_compositions(funcs, max_depth - 1)
                for prev_comp in prev_level:
                    for f in funcs:
                        # используем functools.partial-like closure capture via default args
                        def composed(x, f1=prev_comp, f2=f):
                            return f2(f1(x))
                        compositions.append(composed)
                return prev_level + compositions

            comp_depth = min(3, self.max_depth)
            all_compositions = generate_compositions(functions, comp_depth)

            best = E
            limit = min(100, len(all_compositions))
            for comp in all_compositions[:limit]:
                try:
                    val = comp(E)
                    if val > best and not math.isinf(val):
                        best = val
                except Exception:
                    continue

            if best >= self.overflow_threshold:
                return self.soft_clip(best)
            return best
        except Exception:
            return self.soft_clip(self.overflow_threshold)

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

            n = int(math.floor(x))
            phi = x - n

            analysis = {
                'input': x,
                'function': func_name,
                'result': result,
                'computation_time': compute_time,
                'level_n': n,
                'position_phi': phi,
                'log10_result': math.log10(result) if result > 0 and not math.isinf(result) else -float('inf'),
                'mode': self.mode,
                'is_overflow': result >= self.overflow_threshold,
                'is_infinite': math.isinf(result)
            }

            return analysis

        except Exception as e:
            return {'error': str(e), 'input': x, 'function': func_name}

    def compare_functions(self, x: float) -> Dict[str, Any]:
        """Сравнение всех TRT функций в точке x"""
        functions = ['HCCSF', 'TRANSCEND', 'META_TRANSCEND', 'ULTIMATE_TRANSCEND', 'GOD_TRANSCEND', 'ABSOLUTE', 'UNIVERSUM']

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
            except Exception:
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


# Пример использования (осторожно: тяжёлые расчёты возможны)
if __name__ == "__main__":
    trt = TRT(mode='SECURE', precision=200, cache_limit=10000, overflow_softness=1.0)

    print("=== Анализ TRANSCEND ===")
    analysis = trt.analyze_function(2.0, 'TRANSCEND')
    for key, value in analysis.items():
        print(f"{key}: {value}")

    print("\n=== Сравнение функций при x=1.5 ===")
    comparison = trt.compare_functions(1.5)
    for func, result in comparison.items():
        if 'result' in result:
                    val = result['result']
                    log10v = result.get('log10_result', -float('inf'))
                    print(f"{func}: {val if val < 1e200 else 'very large'} (log10: {log10v:.2f})")

    visualizer = TRTVisualizer(trt)
    visualizer.plot_function('TRANSCEND', 0, 2, 200)
