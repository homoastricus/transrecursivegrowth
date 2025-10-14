import matplotlib.pyplot as plt
import numpy as np
import math


# Реализация функций для HCCSF с защитой от переполнения
def S(t):
    """Сглаживающая функция"""
    if t > 1e100:  # Для очень больших t
        return 1.0
    return t / (1 + t)


def T0(y):
    """Базовый генератор уровня 0"""
    if y >= 0.999:  # Защита от деления на 0
        return 1e100
    return 1 / (1 - y)


def Tn(y, n, max_depth=3):
    """
    Генератор уровня n через рекурсию
    Ограничиваем глубину для избежания переполнения
    """
    if n == 0:
        return T0(y)
    elif n > max_depth:
        # Для высоких уровней используем аппроксимацию
        return 1e100
    else:
        prev = Tn(y, n - 1, max_depth)
        if prev > 700:  # Защита от переполнения exp
            return 1e100
        return math.exp(prev)


def HCCSF(x, max_level=4):
    """
    Функция масштабирования иерархической вычислимой сложности
    x = n + y, где n ∈ ℕ, y ∈ [0,1)
    """
    if x < 0:
        return 0

    n = int(x)  # целая часть
    y = x - n  # дробная часть

    # Ограничиваем максимальный уровень для избежания переполнения
    if n > max_level:
        n = max_level
        y = min(y, 0.99)  # Ограничиваем y для избежания переполнения

    if n == 0:
        return S(T0(y))

    # Вычисляем Tn(y) и Tn(0) с ограничением глубины
    Tn_y = Tn(y, n, max_depth=min(n, 3))
    Tn_0 = Tn(0, n, max_depth=min(n, 3))

    # Применяем формулу HCCSF
    numerator = S(Tn_y) - S(Tn_0)
    denominator = 1 - S(Tn_0)

    if denominator < 1e-10:  # Защита от деления на 0
        return n + 0.999

    return n + numerator / denominator


# Создаем данные для графика с безопасными значениями
x_values = np.arange(0.05, 5.01, 0.01)
y_values = []

# Вычисляем значения с защитой от ошибок
for x in x_values:
    try:
        y_val = HCCSF(x)
        y_values.append(y_val)
    except:
        y_values.append(4.999)  # Значение при ошибке

y_values = np.array(y_values)

# Настройка стиля для научной графики
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))

# Построение графика (оси остаются как было)
ax.plot(x_values, y_values, 'b-', linewidth=2, label='HCCSF(x)')
ax.plot(x_values, y_values, 'ro', markersize=1, alpha=0.3)

# Вертикальные линии на целых значениях (числовая ось)
for n in range(0, 6):
    ax.axvline(x=n, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(n, -0.3, f'n={n}', ha='center', va='top', fontsize=10, color='red')

# ГОРИЗОНТАЛЬНЫЕ ОБЛАСТИ для уровней сложности (слои один над другим)
level_regions = [
    (0, 1, 'L0: Линейный/полиномиальный рост\n(1, 10, 100...)'),
    (1, 2, 'L1: Экспоненциальный рост\n(2ⁿ, 10ⁿ)'),
    (2, 3, 'L2: Двойная экспонента\n(2²ⁿ)'),
    (3, 4, 'L3: Тройная экспонента\n(2²²ⁿ)'),
    (4, 5, 'L4: Тетрация\n(2↑↑n)')
]

colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
for i, (start, end, label) in enumerate(level_regions):
    # Горизонтальные области (слои)
    ax.axhspan(start, end, alpha=0.2, color=colors[i])
    # Подписи уровней слева
    ax.text(-0.3, (start + end) / 2, f'L{i}', ha='right', va='center',
            fontsize=12, color='darkblue', fontweight='bold')
    # Описания уровней справа
    ax.text(5.1, (start + end) / 2, label, ha='left', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.5))

# Горизонтальные линии на уровнях сложности
for level in range(0, 6):
    ax.axhline(y=level, color='green', linestyle='-', alpha=0.3, linewidth=1)

# Подписи осей и заголовок
ax.set_xlabel('Числовая ось (x = n + y)', fontsize=14, fontweight='bold')
ax.set_ylabel('Уровень сложности HCCSF(x)', fontsize=14, fontweight='bold')
ax.set_title('Функция масштабирования иерархической вычислимой сложности (HCCSF)',
             fontsize=16, fontweight='bold', pad=20)

# Сетка
ax.grid(True, alpha=0.3)

# Особые точки и аннотации
special_points = [
    (0.1, 'x=0.1\nT₀≈1.11', 'right'),
    (0.5, 'x=0.5\nT₀=2', 'right'),
    (0.9, 'x=0.9\nT₀=10', 'right'),
    (1.0, 'Переход L0→L1', 'left'),
    (1.5, 'x=1.5\nT₁≈7.39', 'left'),
    (2.0, 'Переход L1→L2', 'left'),
    (2.5, 'x=2.5\nT₂≈1639.4', 'left'),
    (3.0, 'Переход L2→L3', 'left'),
    (4.0, 'Переход L3→L4', 'left')
]

for x, text, ha in special_points:
    try:
        y = HCCSF(x)
        ax.plot(x, y, 'o', color='purple', markersize=6)
        ax.annotate(text, (x, y), xytext=(10 if ha == 'left' else -10, 10),
                    textcoords='offset points', ha=ha, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='purple'),
                    fontsize=9)
    except:
        continue

# Легенда и дополнительные пояснения
ax.legend(loc='upper left', fontsize=12)

explanation_text = '''
HCCSF(x) = n + (S(Tn(y)) - S(Tn(0))) / (1 - S(Tn(0)))
где:
• x = n + y, n in N, y in [0,1)
• T0(y) = 1/(1-y)
• Tn+1(y) = exp(Tn(y))
• S(t) = t/(1+t)

Каждый уровень n соответствует качественному скачку
в вычислительной сложности роста чисел.
Функция непрерывна и строго монотонна.
'''

ax.text(3.0, 2.0, explanation_text, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2),
        verticalalignment='top')

# Настройка осей
ax.set_xlim(-0.1, 5.1)
ax.set_ylim(-0.5, 5.5)

# Показываем поведение функции при приближении к границам уровней
ax.text(0.95, 4.5, 'При y→1: Tn(y)→INF\nHCCSF(x)→n+1',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
ax.text(1.95, 4.3, 'Качественный скачок\nсложности на границах',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

# Добавим стрелки для демонстрации роста
ax.annotate('Рост чисел →', (4.5, -0.2), xytext=(2.5, -0.2),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=12, color='blue', ha='center')

ax.annotate('Рост сложности ↑', (-0.2, 4.5), xytext=(-0.2, 2.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', ha='center', rotation=90)

plt.tight_layout()
plt.savefig('HCCSF_function_plot_final.png', dpi=300, bbox_inches='tight')
plt.show()

# Дополнительная информация о вычисленных значениях
print("\n" + "=" * 70)
print("ХАРАКТЕРНЫЕ ЗНАЧЕНИЯ HCCSF(x)")
print("=" * 70)

test_points = [0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 2.0, 2.1, 2.5, 3.0, 3.5, 4.0, 4.5]
print(f"{'x':>6} {'n':>3} {'y':>6} {'HCCSF(x)':>10} {'Уровень':>8} {'Описание':<25}")
print("-" * 70)

for x in test_points:
    try:
        hccsf_val = HCCSF(x)
        n = int(x)
        y = x - n
        level = int(hccsf_val)
        description = level_regions[level][2].split('\n')[0] if level < len(level_regions) else "Высший уровень"

        print(f"{x:6.2f} {n:3d} {y:6.2f} {hccsf_val:10.4f} {level:>8} {description:<25}")
    except:
        print(f"{x:6.2f} {'-':>3} {'-':>6} {'ERROR':>10} {'-':>8} {'Ошибка вычисления':<25}")

print("=" * 70)
print("График сохранен как 'HCCSF_function_plot_final.png'")
print("Теперь уровни сложности отображаются как горизонтальные слои!")