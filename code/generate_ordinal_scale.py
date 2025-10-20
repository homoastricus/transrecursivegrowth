import matplotlib.pyplot as plt
import numpy as np
from math import radians, sin, cos

# Настройка стиля для гугологического графика
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(25, 25))

# Корректная шкала сверхбыстрых функций (настоящая гугология)
googology_scale = {
    # Базовый уровень - мощные вычислимые функции
    0: "TREE(3)\nДеревья\nКрускала",
    5: "SCG(13)\nМаксимальные\nподграфы",
    10: "SSCG(3)\nСильные\nподграфы",
    15: "BH(9)\nБыстрая\nиерархия\nБучаля",
    20: "Loader(n)\nПрограмма\nЛоадера\nD5(99)",

    # Нотации массивов и ординалов
    25: "BMS\nn(3,3,3)\nМассивы\nБауэрса",
    30: "Y-sequence\nY(1,3)\nЯмашита",
    35: "ψ(Ω_ω)\nОрдинал\nБахмана-Говарда",
    40: "ψ(Ω_Ω_ω)\nРасширенный\nБахман",

    # Достигаем невычислимых областей
    45: "Σ(100)\nBB 100\n~f_ω₁ᶜᴷ(n)",
    50: "Σ(1000)\nBB 1000",
    55: "Σ(10^6)\nМега-BB",
    60: "Σ(Σ(5))\nИтерация\nBB",

    # Системы ординальных обозначений высшего уровня
    65: "Tar(n)\nТарский\nпредел",
    70: "Dimensional\nBMS",
    71: "Y(1,4,7)",
    72: "ψ(I)Недост.",
    73: "ψ(M)",
    74: "ψ(K)",
    75: "ψ(Π₃)",

    # Рекурсивные нотации высшего порядка
    76: "Предел\nΠ₄",
    77: "Σ(Σ(Σ(5)))",
    78: "n(1)\nПервое\nтрансфинитное\nчисло BB",
    79: "Σ^ω(1)",

    # Подходим к Райо
    80: "BB(1000)",
    82: "Σ(10^100)",
    84: "Σ(Rayo(10))",
    86: "Ξ(10^6)",

    # Область Райо
    87: "Rayo(1000)",
    89.5: "Rayo(10^100)",

    # За пределами Райо
    89.8: "Big Foot",
    90: "Sasquatch"
}

# Создание детализированной радиальной сетки
angles = np.linspace(0, 90, 360)
radii = np.linspace(0, 1, 12)

# Отрисовка концентрических окружностей
for r in radii:
    x = r * np.cos(np.radians(angles))
    y = r * np.sin(np.radians(angles))
    ax.plot(x, y, 'w-', alpha=0.15, linewidth=0.5)

# Отрисовка лучей
for angle in range(0, 91, 2):
    rad = radians(angle)
    x = [0, 0.98 * cos(rad)]
    y = [0, 0.98 * sin(rad)]
    alpha = 0.3 if angle % 10 == 0 else 0.15
    ax.plot(x, y, 'w-', alpha=alpha, linewidth=0.3)
counter = 0
# Отрисовка меток гугологических функций
for angle, label in googology_scale.items():
    rad = radians(angle)


    # Циклическая градация расстояний: 0.6, 0.9, 1.2
    distance_levels = [0.82, 0.87, 0.93, 0.98, 1.04]
    distance = distance_levels[counter % 5]
    counter += 1


    x = distance * cos(rad)
    y = distance * sin(rad)

    # Цветовая градация
    if angle <= 30:
        color = '#ffff00'  # Желтый
    elif angle <= 60:
        color = '#ff8000'  # Оранжевый
    elif angle <= 80:
        color = '#ff4000'  # Красно-оранжевый
    elif angle <= 89:
        color = '#ff2000'  # Красный
    else:
        color = '#ff0000'  # Ярко-красный

    # Разный размер шрифта
    fontsize = 6
    if angle in [0, 20, 45, 65, 80, 89.5, 90]:
        fontsize = 7
    if angle in [0, 80, 89.5, 90]:
        fontsize = 8

    rotation = angle - 90
    if angle > 85:
        rotation = angle - 95

    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color=color, rotation=rotation,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.9, edgecolor=color))

# Области гугологической сложности
googology_regions = [
    (0, 25, "Мощные\nкомбинаторные\nфункции", "#ffff00"),
    (25, 45, "Ординальные\nнотации\n(BMS, Y)", "#ffaa00"),
    (45, 65, "Функция\nЗанятого\nБобра", "#ff5500"),
    (65, 80, "Тарский\nи BMS\nсистемы", "#ff2200"),
    (80, 90, "Rayo/FOOT\nАбстрактная\nлогика", "#ff0000")
]

# Отрисовка областей
for start, end, label, color in googology_regions:
    mid_angle = (start + end) / 2
    rad = radians(mid_angle)
    x = 0.66 * cos(rad)
    y = 0.66 * sin(rad)

    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, color=color, weight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.9, edgecolor=color))

# Информационные блоки
info_texts = [
    (12, 0.25, "TREE(3) > Graham's number\nSCG(13) > TREE(3)", "#00ff80"),
    (40, 0.36, "Σ(100) уже невычислима\nна практике", "#ff8040"),
    (75, 0.25, "Tar, BMS, Y-sequence\nиспользуют сложные\nординальные нотации", "#ff4040"),
]

for angle, radius, text, color in info_texts:
    rad = radians(angle)
    x = 1.2*radius * cos(rad)
    y = 1.2*radius * sin(rad)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7, color=color, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))

# Настройка графика
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')

ax.set_title('КОРРЕКТНАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА ГУГОЛОГИЧЕСКИХ ФУНКЦИЙ',
             fontsize=14, pad=20, color='#ff4444', weight='bold')

plt.tight_layout()
plt.show()

# Сравнительная таблица с правильными величинами
print("\n" + "=" * 80)
print("КОРРЕКТНАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА ГУГОЛОГИЧЕСКИХ ФУНКЦИЙ:")
print("=" * 80)
print(f"{'Угол':<6} {'Функция':<20} {'Характеристика':<45}")
print("-" * 80)
correct_comparison = [
    (0, "TREE(3)", "Знаменитое комбинаторное число > G(64)"),
    (5, "SCG(13)", "Максимальные подграфы > TREE(3)"),
    (20, "Loader(n)", "D5(99) - мощная вычислимая функция"),
    (25, "BMS", "Массивы Бауэрса с ординалами"),
    (30, "Y-sequence", "Нотация Ямашиты"),
    (45, "Σ(100)", "Невычислимая на практике"),
    (65, "Tar(n)", "Тарский предел ординальных нотаций"),
    (70, "Dimensional BMS", "4-мерные массивы Бауэрса"),
    (80, "BB(1000)", "Как requested"),
    (89.5, "Rayo(10^100)", "Как requested - теория множеств"),
    (90, "Sasquatch", "Один из крупнейших определённых чисел")
]

for angle, func, desc in correct_comparison:
    print(f"{angle:>5}° {func:<20} {desc:<45}")
print("=" * 80)