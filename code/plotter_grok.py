import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Callable, Literal, Tuple, Optional, Any
from decimal import Decimal, getcontext
from functools import lru_cache
import warnings


class TRT:
    """Класс для работы с Трансрекурсивной Теорией Роста"""

    def __init__(self, mode: Literal['FULL', 'SECURE'] = 'FULL', precision: int = 1000):
        self.mode = mode
        self.precision = precision
        getcontext().prec = precision

        if self.mode == 'FULL':
            self._init_full_mode()
        else:
            self._init_secure_mode()

        self._clear_caches()

        self.complexity_levels = {
            0: "Линейный/полиномиальный рост",
            1: "Экспоненциальный рост",
            2: "Двойная экспонента",
            3: "Тройная экспонента",
            4: "Тетрация",
            5: "Пентация",
            6: "Гипероператоры высших порядков"
        }

    def _clear_caches(self):
        """Очистка кэшей между вычислениями"""
        self._hccsf_cache = {}
        self._t_n_cache = {}

    def _init_full_mode(self):
        self.max_depth = 100  # Ограничить для избежания переполнения
        self.max_level = 100
        self.overflow_threshold = 1e308  # Максимум для float64
        self.max_iterations = 1000
        self.max_meta_iterations = 1000

    def _init_secure_mode(self):
        self.max_depth = 20
        self.max_level = 30
        self.overflow_threshold = 1e100
        self.max_iterations = 100
        self.max_meta_iterations = 100

    def S(self, t: float) -> float:
        """Сглаживающая функция S(t) = t/(1+t)"""
        if t > self.overflow_threshold:
            return 1.0
        if t < 0:
            return 0.0
        return t / (1 + t)

    def T0(self, phi: float) -> float:
        """Базовый генератор уровня 0: T₀(φ) = 1/(1-φ)"""
        if phi >= 1.0:
            return self.overflow_threshold
        if phi <= 0.0:
            return 1.0

        if self.mode == 'SECURE' and phi >= 0.999:
            return self.overflow_threshold

        return 1.0 / (1.0 - phi)

    def Tn(self, phi: float, n: int) -> float:
        """Рекурсивный генератор уровня n: Tₙ(φ) = exp(Tₙ₋₁(φ))"""
        if n < 0:
            return 0.0
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
            else:
                result = math.exp(prev)

            self._t_n_cache[cache_key] = result
            return result

        except (OverflowError, ValueError):
            return self.overflow_threshold

    def HCCSF(self, n: int, phi: float) -> float:
        """Функция масштабирования иерархической вычислимой сложности"""
        if n < 0:
            return 0.0

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

                if hccsf_val >= 7000:  # exp(700) уже слишком большое
                    return self.overflow_threshold

                current = math.exp(hccsf_val)

            except (OverflowError, ValueError):
                return self.overflow_threshold

        return current

    def TRANSCEND(self, x: float) -> float:
        """Трансцендентная функция роста"""
        self._clear_caches()  # Очищаем кэш для нового вычисления

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
            if exp_arg < -7000:  # exp(-700) практически 0
                inner = 1.0
            else:
                inner = 1 - math.exp(exp_arg)

            G = self.HCCSF(n, inner)
        except (OverflowError, ValueError):
            return float('inf')

        # Вычисляем k
        try:
            if G >= 7000:
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
            # ВНИМАНИЕ: Это самая взрывоопасная часть!
            power_base = meta_result
            power_exp = T_prev

            # Защита от переполнения
            if power_base <= 0:
                power_result = 0
            elif power_exp * math.log(power_base) > 7000:  # exp(700) → overflow
                return float('inf')
            else:
                power_result = power_base ** power_exp

            if power_result > 7000:  # exp(700) → overflow
                return float('inf')

            result = math.exp(power_result)
            return result

        except (OverflowError, ValueError):
            return float('inf')


def plot_transcend(a: float, b: float, num_points: int = 1000) -> None:
    """
    Построение графика функции TRANSCEND на интервале [a, b]
    """
    trt = TRT(mode='SECURE', precision=1000)  # Используем SECURE для стабильности

    x_values = np.linspace(a, b, num_points)
    y_values = []
    log_y_values = []

    print(f"Вычисление TRANSCEND на интервале [{a}, {b}]...")

    for i, x in enumerate(x_values):
        if i % 100 == 0:
            print(f"Прогресс: {i}/{num_points}")

        try:
            transcend_val = trt.TRANSCEND(x)

            if transcend_val == float('inf'):
                y_values.append(np.nan)
                log_y_values.append(np.nan)
            else:
                y_values.append(transcend_val)
                if transcend_val > 0:
                    log_y_values.append(math.log(transcend_val))
                else:
                    log_y_values.append(np.nan)

        except Exception as e:
            print(f"Ошибка при x={x}: {e}")
            y_values.append(np.nan)
            log_y_values.append(np.nan)

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # График 1: TRANSCEND(x)
    ax1.plot(x_values, y_values, 'b-', linewidth=2, label='TRANSCEND(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('TRANSCEND(x)')
    ax1.set_title(f'Функция TRANSCEND(x) на интервале [{a}, {b}]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')  # Логарифмическая шкала для y

    # График 2: ln(TRANSCEND(x))
    ax2.plot(x_values, log_y_values, 'r-', linewidth=2, label='ln(TRANSCEND(x))')
    ax2.set_xlabel('x')
    ax2.set_ylabel('ln(TRANSCEND(x))')
    ax2.set_title(f'Натуральный логарифм TRANSCEND(x) на интервале [{a}, {b}]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Статистика
    valid_values = [v for v in y_values if not np.isnan(v) and v != float('inf')]
    if valid_values:
        print(f"\nСтатистика:")
        print(f"Минимум: {min(valid_values):.2e}")
        print(f"Максимум: {max(valid_values):.2e}")
        print(f"Диапазон: {max(valid_values) / min(valid_values):.2e}")


# Пример использования
if __name__ == "__main__":
    # Тестируем на безопасном интервале
    plot_transcend(a=0, b=0.8, num_points=500)