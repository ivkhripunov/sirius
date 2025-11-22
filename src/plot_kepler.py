import matplotlib.pyplot as plt
import numpy as np


def plot_kepler(kepler_array, times, figsize=(15, 10), dpi=300):
    elements = ['a', 'e', 'i', 'ω', 'Ω', 'ν']
    units = ['м', '', 'рад', 'рад', 'рад', 'рад']
    full_names = [
        'Большая полуось',
        'Эксцентриситет',
        'Наклонение',
        'Аргумент перицентра',
        'Долгота восходящего узла',
        'Истинная аномалия'
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    for elem_idx in range(6):
        ax = axes[elem_idx]

        ax.plot(times, kepler_array[:, elem_idx],
                linewidth=1.5,
                alpha=0.8,)

        ax.set_xlabel('Время', fontsize=10)
        ax.set_ylabel(f'{elements[elem_idx]} [{units[elem_idx]}]', fontsize=10)
        ax.set_title(full_names[elem_idx], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

        if kepler_array.shape[0] > 1 and elem_idx == 0:
            ax.legend(fontsize=8, loc='best')

    plt.suptitle('Эволюция кеплеровых элементов орбиты',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    fig.savefig('kepler_evolution.png')
