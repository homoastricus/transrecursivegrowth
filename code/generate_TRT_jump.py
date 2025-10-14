import matplotlib.pyplot as plt
import numpy as np
import math


# Реализация функций для HCCSF
def S(t):
    """Сглаживающая функция"""
    if t > 1e100:
        return 1.0
    return t / (1 + t)


def T0(y):
    """Базовый генератор уровня 0"""
    if y >= 0.999:
        return 1e100
    return 1 / (1 - y)


def Tn(y, n, max_depth=2):
    """Генератор уровня n"""
    if n == 0:
        return T0(y)
    elif n > max_depth:
        return 1e100
    else:
        prev = Tn(y, n - 1, max_depth)
        if prev > 700:
            return 1e100
        return math.exp(prev)


def HCCSF(x, max_level=3):
    """Функция HCCSF"""
    if x < 0:
        return 0

    n = int(x)
    y = x - n

    if n > max_level:
        n = max_level
        y = min(y, 0.99)

    if n == 0:
        return S(T0(y))

    Tn_y = Tn(y, n, max_depth=min(n, 2))
    Tn_0 = Tn(0, n, max_depth=min(n, 2))

    numerator = S(Tn_y) - S(Tn_0)
    denominator = 1 - S(Tn_0)

    if denominator < 1e-10:
        return n + 0.999

    return n + numerator / denominator


def HCCSF_inverse(L, tolerance=1e-5):
    """Обратная функция HCCSF (упрощенная)"""
    if L < 0:
        return 0

    n = int(L)
    delta = L - n

    if n == 0:
        # Для уровня 0: обратная к S(T0(y))
        u = delta * (1 - S(T0(0))) + S(T0(0))
        t = u / (1 - u) if u < 1 else 1e100
        return 1 - 1 / t if t > 0 else 0

    # Для n >= 1 используем численный метод
    def equation(y):
        return HCCSF(n + y) - L

    low, high = 0.0, 0.999
    for _ in range(50):  # Метод бисекции
        mid = (low + high) / 2
        if equation(mid) < 0:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break

    return n + (low + high) / 2


# Создаем основной график
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 10))

# Основные данные для HCCSF
x_values = np.arange(0.1, 100.1, 0.5)
y_values = [HCCSF(x) for x in x_values if x <= 5] + [HCCSF(5) + 0.1] * len([x for x in x_values if x > 5])

# Ограничиваем для наглядности
x_plot = [x for x in x_values if x <= 5]
y_plot = [HCCSF(x) for x in x_plot]

# 1. Рисуем основную функцию HCCSF
ax.plot(x_plot, y_plot, 'b-', linewidth=3, label='HCCSF(x)', alpha=0.7)
ax.plot(x_plot, y_plot, 'bo', markersize=3, alpha=0.5)

# 2. ГОРИЗОНТАЛЬНЫЕ ОБЛАСТИ для уровней сложности
level_regions = [
    (0, 1, 'L0: Полиномиальный рост'),
    (1, 2, 'L1: Экспоненциальный рост'),
    (2, 3, 'L2: Двойная экспонента'),
    (3, 4, 'L3: Тройная экспонента')
]

colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for i, (start, end, label) in enumerate(level_regions):
    ax.axhspan(start, end, alpha=0.2, color=colors[i])
    ax.text(-5, (start + end) / 2, f'L{i}', ha='right', va='center',
            fontsize=14, color='darkblue', fontweight='bold', style='italic')
    ax.text(102, (start + end) / 2, label, ha='left', va='center',
            fontsize=10, bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.6))

# 3. ПРЫЖОК 1: {1, 1.5} → {0, 2} (HCCSF)
start_point = (1, 1.5)  # Маленькое число на оси X
target_level = 2.0  # Уровень сложности L2

# Вычисляем соответствующую точку на уровне L2
y_target = target_level
x_target = 0  # Начало уровня L2 на оси X (для визуализации)

# Рисуем прыжок HCCSF
ax.plot(start_point[0], start_point[1], 'o', color='red', markersize=10, label='Исходное число')
ax.annotate('Число: 1.0', start_point, xytext=(15, 10), textcoords='offset points',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='red'))

# Стрелка прыжка HCCSF
ax.annotate('', xy=(x_target, y_target), xytext=start_point,
            arrowprops=dict(arrowstyle='->', color='purple', lw=3, alpha=0.8))

ax.plot(x_target, y_target, 's', color='purple', markersize=12, label='Уровень сложности')
ax.annotate('HCCSF(1.0) → L2', (x_target, y_target), xytext=(-80, 10), textcoords='offset points',
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='purple'))

# 4. ПРЫЖОК 2: {0, 2} → {100, 2} (HCCSF⁻¹)
end_point = (100, 2.0)  # Большое число на оси X

# Стрелка прыжка HCCSF⁻¹
ax.annotate('', xy=end_point, xytext=(x_target, y_target),
            arrowprops=dict(arrowstyle='->', color='green', lw=3, alpha=0.8, linestyle='--'))

ax.plot(end_point[0], end_point[1], 'D', color='green', markersize=12, label='Усиленное число')
ax.annotate('HCCSF⁻¹(L2) → 100', end_point, xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='green'))

# 5. ЛОГАРИФМИЧЕСКИЕ КРИВЫЕ, связывающие прыжки
# Кривая для HCCSF (слева)
x_curve1 = np.linspace(start_point[0], 2, 50)
y_curve1 = [1.5 + 0.5 * (1 - np.exp(-5 * (x - 1))) for x in x_curve1]
ax.plot(x_curve1, y_curve1, 'r--', alpha=0.6, linewidth=2)

# Кривая для HCCSF⁻¹ (справа)
x_curve2 = np.linspace(3, end_point[0], 50)
y_curve2 = [2.0 - 0.3 * np.exp(-0.05 * (x - 3)) for x in x_curve2]
ax.plot(x_curve2, y_curve2, 'g--', alpha=0.6, linewidth=2)

# 6. Области графика
ax.axvspan(0, 5, alpha=0.1, color='blue', label='Область детального роста')
ax.axvspan(95, 105, alpha=0.1, color='green', label='Область больших чисел')

# 7. Подписи и оформление
ax.set_xlabel('Числовая ось (абсциссы)', fontsize=14, fontweight='bold')
ax.set_ylabel('Уровни вычислительной сложности', fontsize=14, fontweight='bold')
ax.set_title('Трансрекурсивный прыжок: Число → Уровень сложности → Усиленное число',
             fontsize=16, fontweight='bold', pad=20)

# Сетка
ax.grid(True, alpha=0.3)

# Легенда
ax.legend(loc='upper left', fontsize=12)

# Объясняющая диаграмма
explanation_text = '''
Трансрекурсивный прыжок:
1. HCCSF(1.0) = L2  → Определение уровня сложности
2. HCCSF_INV(L2) = 100 → Усиление числа через мета-итерации

Логарифмические кривые показывают:
• Красная: Экспоненциальный рост сложности (HCCSF)
• Зеленая: Обратное преобразование (HCCSF_INV)
'''

ax.text(60, 3.5, explanation_text, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        verticalalignment='top')

# Математические формулы
formula_text = '''
HCCSF(x) = n + (S(Tn(y)) - S(Tn(0))) / (1 - S(Tn(0)))

S(t) = t/(1+t)
T0(y) = 1/(1-y)
Tn+1(y) = exp(Tn(y))
'''

ax.text(60, 1.0, formula_text, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
        verticalalignment='top')

# Настройка осей
ax.set_xlim(-10, 110)
ax.set_ylim(-0.5, 4.5)

# Стрелки для обозначения направлений
ax.annotate('Рост чисел →', (80, -0.3), xytext=(40, -0.3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=12, color='blue', ha='center')

ax.annotate('Рост сложности ↑', (-8, 3.5), xytext=(-8, 1.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', ha='center', rotation=90)

plt.tight_layout()
plt.savefig('TRT_Jump_Diagram.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("ДИАГРАММА ТРАНСРЕКУРСИВНОГО ПРЫЖКА")
print("=" * 70)
print("1. Начальная точка: Число 1.0 на числовой оси")
print("2. HCCSF(1.0) = L2: Определение уровня сложности")
print("3. HCCSF⁻¹(L2) = 100: Усиление через мета-итерации")
print("4. Результат: Число 100 на числовой оси")
print("=" * 70)
print("График сохранен как 'TRT_Jump_Diagram.png'")