import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy.stats import gaussian_kde
from scipy import integrate

# Константы
N = 100  # число частиц
radius = 0.02  # радиус частиц
mass = 1.0  # молекулярная масса
box_size = 1.0  # размер контейнера
dt = 0.02  # шаг времени
k_B = 0.01  # постоянная Больцмана

# Начальная температура
initial_temperature = 300.0

def get_mean_speed(T):
    """Средняя скорость по распределению Максвелла"""
    return np.sqrt(np.pi * k_B * T / (2 * mass))  # Для 2D случая

def maxwell_2d(v, T):
    """Двумерное распределение Максвелла по скоростям"""
    return (4*np.pi * (mass/(2*np.pi*k_B*T))**(3/2) * v**2 * np.exp(-mass*v**2/(2*k_B*T)))

# Инициализация частиц
def initialize_particles():
    np.random.seed(42)
    positions = np.random.rand(N, 2) * (box_size - 2 * radius) + radius
    angles = np.random.rand(N) * 2 * np.pi
    speeds = np.random.normal(loc=get_mean_speed(initial_temperature),
                             scale=0.1*get_mean_speed(initial_temperature),
                             size=N)
    velocities = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, None]
    return positions, velocities

positions, velocities = initialize_particles()

def handle_collisions():
    """Обработка столкновений частиц между собой и со стенками"""
    for i in range(N):
        # Со стенками
        for dim in [0, 1]:
            if positions[i, dim] < radius:
                positions[i, dim] = radius
                velocities[i, dim] = abs(velocities[i, dim])
            elif positions[i, dim] > box_size - radius:
                positions[i, dim] = box_size - radius
                velocities[i, dim] = -abs(velocities[i, dim])
        
        # С другими частицами
        for j in range(i+1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2 * radius:
                # Коррекция позиций
                direction = (positions[i] - positions[j]) / dist
                positions[i] += (2*radius - dist)/2 * direction
                positions[j] -= (2*radius - dist)/2 * direction
                
                # Обмен импульсами
                dv = velocities[i] - velocities[j]
                impulse = np.dot(dv, direction) * direction
                velocities[i] -= impulse
                velocities[j] += impulse

def update_positions():
    """Обновление позиций частиц"""
    global positions
    positions += velocities * dt
    handle_collisions()

# Настройка графики
plt.rcParams['figure.dpi'] = 100
fig = plt.figure(figsize=(12, 5))
ax_particles = fig.add_subplot(121, aspect='equal')
ax_hist = fig.add_subplot(122)

ax_particles.set_xlim(0, box_size)
ax_particles.set_ylim(0, box_size)
ax_particles.set_title("Движение молекул")
ax_hist.set_title("Распределение скоростей Максвелла")
ax_hist.set_xlabel("Скорость, v")
ax_hist.set_ylabel("Количество частиц")

# Фиксированные пределы для графика скорости
FIXED_MAX_SPEED = 9.0
ax_hist.set_xlim(0,FIXED_MAX_SPEED)
ax_hist.set_ylim(0, 60)  # Фиксированный предел по Y

# Рисуем частицы как круги
particles = [plt.Circle((0,0), radius, color='blue', alpha=0.7) for _ in range(N)]
for p in particles:
    ax_particles.add_patch(p)

# Линии для распределения скоростей
speed_line, = ax_hist.plot([], [], 'b-', lw=2, label='Симуляция')
theory_line, = ax_hist.plot([], [], 'r--', lw=2, label='Теория')
ax_hist.legend()

# Слайдер температуры
slider_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
temp_slider = Slider(slider_ax, 'Температура (K)', 100, 500, 
                    valinit=initial_temperature, valstep=10)

def update_temperature(val):
    """Масштабирование скоростей при изменении температуры"""
    T = temp_slider.val
    current_kinetic = 0.5 * mass * np.sum(velocities**2)
    target_kinetic = N * k_B * T  # Для 2D: 2 степени свободы → kT на частицу
    scale = np.sqrt(target_kinetic / current_kinetic) if current_kinetic > 0 else 1
    velocities[:] *= scale

temp_slider.on_changed(update_temperature)

def init():
    """Инициализация анимации"""
    for p in particles:
        p.set_center((0,0))
    speed_line.set_data([], [])
    theory_line.set_data([], [])
    return particles + [speed_line, theory_line]

def animate(i):
    """Функция анимации"""
    update_positions()
    
    # Обновляем частицы
    for j, p in enumerate(particles):
        p.set_center(positions[j])
    
    # Обновляем график скоростей
    speeds = np.linalg.norm(velocities, axis=1)
    if len(speeds) > 1:
        T = temp_slider.val
        x = np.linspace(0, FIXED_MAX_SPEED, 200)
        
        # Эмпирическое распределение (KDE)
        kde = gaussian_kde(speeds)
        y_kde = kde(x)
        
        # Теоретическое распределение
        y_theory = maxwell_2d(x, T)
        
        # Нормировка для сохранения площади под кривой
        y_kde = y_kde / integrate.trapezoid(y_kde, x) * N
        y_theory = y_theory / integrate.trapezoid(y_theory, x) * N
        
        # Обновление графиков
        speed_line.set_data(x, y_kde)
        theory_line.set_data(x, y_theory)
    
    return particles + [speed_line, theory_line]

# Запуск анимации
ani = animation.FuncAnimation(
    fig, animate, frames=200, init_func=init,
    interval=20, blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()
