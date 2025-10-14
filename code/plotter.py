import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Tuple
import warnings


class TRTPlotter:
    def __init__(self, mode: str = 'SECURE', precision: int = 1000):
        self.mode = mode
        self.precision = precision

        # Инициализация ограничений
        if self.mode == 'SECURE':
            self.max_value = 1e100  # Максимальное значение для отображения
            self.max_depth = 20
        else:
            self.max_value = float('inf')
            self.max_depth = float('inf')

    def S(self, t: float) -> float:
        """Сглаживающая функция S(t) = t/(1+t)"""
        if t > 1e100:
            return 1.0
        return t / (1 + t)

    def T0(self, phi: float) -> float:
        """Базовый генератор уровня 0"""
        if phi >= 0.9999999999:
            return self.max_value
        return 1.0 / (1.0 - phi)

    def Tn(self, phi: float, n: int) -> float:
        """Рекурсивный генератор уровня n"""
        if n == 0:
            return self.T0(phi)
        if n > self.max_depth:
            return self.max_value

        prev = self.Tn(phi, n - 1)
        if prev > 1000:
            return self.max_value
        return math.exp(prev)

    def HCCSF(self, n: int, phi: float) -> float:
        """Функция масштабирования иерархической вычислимой сложности"""
        if n < 0:
            return 0.0

        if n == 0:
            return self.S(self.T0(phi))

        try:
            Tn_phi = self.Tn(phi, n)
            Tn_0 = self.Tn(0.0, n)

            S_Tn_phi = self.S(Tn_phi)
            S_Tn_0 = self.S(Tn_0)

            numerator = S_Tn_phi - S_Tn_0
            denominator = 1 - S_Tn_0

            if abs(denominator) < 1e-300:
                return n + 0.999999999
            return n + numerator / denominator

        except (OverflowError, ZeroDivisionError):
            return n + 0.999999999

    def META_ITER(self, G: float, k: int) -> float:
        """Оператор мета-итерации"""
        if k == 0:
            return G

        current = G
        max_iter = min(k, 100) if self.mode == 'SECURE' else k

        for i in range(max_iter):
            try:
                if current > self.max_value:
                    return self.max_value

                inner_arg = 1 - math.exp(-current)
                n_level = int(math.floor(G))
                hccsf_val = self.HCCSF(n_level, inner_arg)
                current = math.exp(hccsf_val)

            except (OverflowError, ValueError):
                return self.max_value

        return current

    def TRANSCEND(self, x: float) -> float:
        """Трансцендентная функция роста"""
        n = int(x)
        phi = x - n

        # Для n=0 используем T0(phi)
        if n == 0:
            T_prev = self.T0(phi)
        else:
            T_prev = self.Tn(phi, n - 1)

        # Ограничение для избежания переполнения
        if T_prev > 1000 and self.mode == 'SECURE':
            return self.max_value

        try:
            G = self.HCCSF(n, 1 - math.exp(-T_prev))
            k = int(math.exp(G))

            if k > 1000 and self.mode == 'SECURE':
                k = 1000

            meta_result = self.META_ITER(G, k)

            # Вычисление степени с проверкой переполнения
            power_base = min(meta_result, 1000) if self.mode == 'SECURE' else meta_result
            exponent = min(T_prev, 100) if self.mode == 'SECURE' else T_prev

            power_result = power_base ** exponent
            final_result = math.exp(power_result)

            return min(final_result, self.max_value)

        except (OverflowError, ValueError):
            return self.max_value

    def safe_log10(self, value: float) -> float:
        """Безопасное вычисление log10 с обработкой больших значений"""
        if value <= 0:
            return 0
        if value > 1e100:
            return 100
        try:
            return math.log10(value)
        except (OverflowError, ValueError):
            return 100

    def generate_plot_data(self, start: float, end: float, num_points: int = 1000) -> Tuple[List[float], List[float]]:
        """Генерация данных для графика"""
        x_values = np.linspace(start, end, num_points)
        y_values = []

        print(f"Генерация графика TRANSCEND(x) на интервале [{start}, {end}]...")

        for i, x in enumerate(x_values):
            if i % 100 == 0:
                print(f"Вычислено {i}/{num_points} точек...")

            try:
                y = self.TRANSCEND(x)
                #log_y = self.safe_log10(y)
                y_values.append(y)
            except:
                y_values.append(100)  # Максимальное значение для логарифма

        return x_values.tolist(), y_values

    def plot_transcend(self, start: float, end: float, num_points: int = 1000,
                       title: str = "Функция TRANSCEND(x)",
                       save_path: str = None):
        """Построение графика функции TRANSCEND"""

        x_values, y_values = self.generate_plot_data(start, end, num_points)

        plt.figure(figsize=(12, 8))
        plt.plot(x_values, y_values, 'b-', linewidth=2, label='log₁₀(TRANSCEND(x))')

        # Настройка графика
        plt.xlabel('x', fontsize=12)
        plt.ylabel('log₁₀(TRANSCEND(x))', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # Добавление пояснений
        plt.text(0.02, 0.98, f'Режим: {self.mode}', transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Автоматическое масштабирование
        plt.ylim(bottom=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен как {save_path}")

        plt.tight_layout()
        plt.show()

        # Вывод статистики
        self.print_statistics(x_values, y_values)

    def print_statistics(self, x_values: List[float], y_values: List[float]):
        """Вывод статистики по данным графика"""
        max_y = max(y_values)
        max_x = x_values[y_values.index(max_y)]

        print(f"\n📊 Статистика графика:")
        print(f"   • Диапазон x: [{x_values[0]:.3f}, {x_values[-1]:.3f}]")
        print(f"   • Максимальное значение: log₁₀(TRANSCEND({max_x:.3f})) = {max_y:.2f}")
        print(f"   • Примерное значение TRANSCEND: 10^{max_y:.0f}")
        print(f"   • Количество точек: {len(x_values)}")

        # Анализ критических точек
        critical_points = []
        for i in range(1, len(y_values) - 1):
            if y_values[i] > y_values[i - 1] and y_values[i] > y_values[i + 1]:
                critical_points.append((x_values[i], y_values[i]))

        if critical_points:
            print(f"   • Критические точки роста: {len(critical_points)}")
            for x, y in critical_points[:3]:  # Показываем первые 3
                print(f"     x={x:.3f}, log₁₀(TRANSCEND)={y:.2f}")


# Примеры использования
def demo_plots():
    """Демонстрационные графики"""

    # 1. График на малом интервале [0, 2]
    print("=" * 50)
    print("ГРАФИК 1: TRANSCEND(x) на [0, 2]")
    print("=" * 50)

    plotter1 = TRTPlotter(mode='SECURE')
    plotter1.plot_transcend(0, 2, 500,
                            "TRANSCEND(x) на интервале [0, 2]",
                            "transcend_0_2.png")

    # 2. График на интервале [0, 1] с высокой детализацией
    print("\n" + "=" * 50)
    print("ГРАФИК 2: TRANSCEND(x) на [0, 1] (высокая детализация)")
    print("=" * 50)

    plotter2 = TRTPlotter(mode='SECURE')
    plotter2.plot_transcend(0, 1, 1000,
                            "TRANSCEND(x) на интервале [0, 1]",
                            "transcend_0_1_detailed.png")

    # 3. График на интервале [1, 3]
    print("\n" + "=" * 50)
    print("ГРАФИК 3: TRANSCEND(x) на [1, 3]")
    print("=" * 50)

    plotter3 = TRTPlotter(mode='SECURE')
    plotter3.plot_transcend(1, 3, 500,
                            "TRANSCEND(x) на интервале [1, 3]",
                            "transcend_1_3.png")


# Функция для построения произвольного графика
def plot_custom_range(start: float, end: float, num_points: int = 1000, mode: str = 'SECURE'):
    """Построение графика на произвольном интервале"""

    plotter = TRTPlotter(mode=mode)
    title = f"TRANSCEND(x) на интервале [{start}, {end}]"
    filename = f"transcend_{start}_{end}.png".replace('.', '_')

    plotter.plot_transcend(start, end, num_points, title, filename)


# Запуск демонстрации
if __name__ == "__main__":
    # Демонстрационные графики
    #demo_plots()

    # Пример построения произвольного графика
    plot_custom_range(1, 1.3, 800, 'SECURE')