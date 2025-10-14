import matplotlib.pyplot as plt
import numpy as np
import math

# Настройка стиля для научной графики
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))

# Настройка осей
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)

# Подписи осей
ax.set_xlabel('Числовая ось (абсциссы)', fontsize=14, fontweight='bold')
ax.set_ylabel('Уровни вычислительной сложности', fontsize=14, fontweight='bold')
ax.set_title('Принцип работы HCCSF в Трансрекурсивной Теории (TRT)',
             fontsize=16, fontweight='bold', pad=20)

# Основные уровни сложности
levels = [1, 2, 3, 4, 5, 6]
level_labels = ['L1', 'L2', 'L3', 'L4', 'L5', 'Ln']

# Рисуем уровни сложности горизонтальными линиями
for i, level in enumerate(levels):
    ax.axhline(y=level, color='blue', linestyle='-', alpha=0.3)
    ax.text(-0.5, level, level_labels[i], ha='right', va='center',
            fontsize=12, fontweight='bold', color='darkblue')

# Подписи для уровней сложности (примеры функций)
level_examples = {
    1: 'Экспоненты\n$2^n, 10^n$',
    2: 'Двойные экспоненты\n$2^{2^n}$',
    3: 'Тройные экспоненты\n$2^{2^{2^n}}$',
    4: 'Тетрация\n$2↑↑n$',
    5: 'Пентация\n$2↑↑↑n$',
    6: 'Гипероператоры\nвысших порядков'
}

for level, text in level_examples.items():
    ax.text(8.2, level, text, ha='left', va='center',
            fontsize=9, style='italic', color='green')

# Числовая ось с основными делениями
main_ticks = [1, 2, 3, 4, 5, 6, 7]
ax.set_xticks(main_ticks)
ax.set_xticklabels(['1', '2', '3', '4', '5', 'n', 'n+1'], fontsize=12)

# Сублинейная шкала между n и n+1 (демонстрация сжатия)
n = 6
n_plus_1 = 7

# Позиции субделений (демонстрирующие сжатие)
sub_divisions = [0.5, 0.75, 0.88, 0.94, 0.97, 0.985]
sub_positions = [n + x for x in sub_divisions]

# Рисуем субделения
for pos in sub_positions:
    ax.axvline(x=pos, ymin=0, ymax=0.1, color='red', linestyle='--', alpha=0.7)

# Подписи для субделений
sub_labels = ['0.5', '0.75', '0.88', '0.94', '0.97', '0.985']
for pos, label in zip(sub_positions, sub_labels):
    ax.text(pos, -0.2, label, ha='center', va='top', fontsize=8, color='red')

# Демонстрация одного цикла HCCSF
# 1. Начальная точка (число около 1)
start_x = 1.2
start_y = 0.2
ax.plot(start_x, start_y, 'ro', markersize=8, label='Начальное число')

# 2. Прыжок на уровень сложности L1
ax.arrow(start_x, start_y, 0, 1 - start_y,
         head_width=0.1, head_length=0.1, fc='purple', ec='purple', linestyle='-')
ax.plot(start_x, 1, 'o', color='orange', markersize=8, label='Уровень сложности')

# 3. Прыжок на числовую ось (определение нового n)
mid_x = 3.5
ax.arrow(start_x, 1, mid_x - start_x, 0,
         head_width=0.1, head_length=0.1, fc='blue', ec='blue', linestyle='--')
ax.plot(mid_x, 1, 's', color='blue', markersize=8, label='Определение уровня')

# 4. Прыжок на уровень сложности Ln
target_level = 4
ax.arrow(mid_x, 1, 0, target_level - 1,
         head_width=0.1, head_length=0.1, fc='green', ec='green', linestyle='-')
ax.plot(mid_x, target_level, 'D', color='green', markersize=10, label='Новый уровень сложности')

# 5. Обратное отображение на числовую ось
final_x = 6.8
ax.arrow(mid_x, target_level, final_x - mid_x, 0,
         head_width=0.1, head_length=0.1, fc='red', ec='red', linestyle=':')
ax.plot(final_x, target_level, 'o', color='red', markersize=8, label='Результат')

# Аннотации для объяснения процесса
annotations = [
    (start_x, start_y, 'Число\n(вход)', 'black'),
    (start_x, 1, 'HCCSF(x)\nОпределение\nуровня', 'orange'),
    (mid_x, 1, 'HCCSF⁻¹(L)\nОбратное\nотображение', 'blue'),
    (mid_x, target_level, 'Новый уровень\nсложности', 'green'),
    (final_x, target_level, 'Усиленное\nчисло', 'red')
]

for x, y, text, color in annotations:
    ax.annotate(text, (x, y), xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color=color),
                fontsize=9, ha='left')

# Легенда
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=11)

# Дополнительные пояснения
explanation_text = '''
Принцип работы HCCSF:
1. Число → Уровень сложности (HCCSF)
2. Уровень → Новое число (HCCSF⁻¹)  
3. Усиление через мета-итерации
4. Результат → Новый уровень сложности

Сублинейная шкала показывает сжатие числовой оси
между уровнями n и n+1, демонстрируя экспоненциальный
рост сложности внутри каждого уровня.
'''

ax.text(2.5, 6.5, explanation_text, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        verticalalignment='top')

# Сетка и оформление
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('TRT_HCCSF_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

print("График сохранен как 'TRT_HCCSF_diagram.png'")
print("Размер: 14x10 дюймов, разрешение: 300 DPI - подходит для научной статьи")