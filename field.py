from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lina
from scipy.constants import pi, epsilon_0

K = 1 / (4 * pi * epsilon_0)
POWER_COLOR = "black"
V_COLOR = "black"


def create_graf(q_arr: list = None, x0: float = 0, y0: float = 0, x1: float = 10, y1: float = 10,
                acc_power: int = 100, acc_charges: int = 100, freq: float = 2.0,
                circle: bool = False, arrows: bool = False):
    """Отрисовка графика

    * `q_arr` - массив зарядов вида `[[q, x1, y1, x2, y2] ...]`.
    * `x0` - минимальная абсцисса графика.
    * `y0` - минимальная ордината графика.
    * `x1` - максимальная абсцисса графика.
    * `y1` - максимальная ордината графика.
    * `acc_power` - точность отрисовки силовых линий.
    * `acc_charges` - точность отрисовки линейных зарядов.
    * `freq` - частота отрисовки линий.
    * `circle` - режим отрисовки эквипотенциалов.
    * `arrows` - режим отрисовки силовых линий.
    """

    if q_arr is None:
        q_arr = []

    fig = plt.Figure()
    ax = fig.add_subplot(111)

    mesh = create_mesh(tuple([x0, x1]), tuple([y0, y1]), tuple([acc_power, acc_power]))
    fig, ax = plot_axis(mesh, figax=tuple([fig, ax]))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    if len(q_arr):
        qtopos = np.vstack(q_arr)
        E = make_vector_field(qtopos, acc_charges)

        if circle:
            plot_potential(q_arr, mesh, tuple([fig, ax]), acc_charges, freq)  # Отрисовка поля эквипотенциалов

        if arrows:
            plot_vector_field(mesh, E(mesh), freq, tuple([fig, ax]))  # Отрисовка поля напряжённости

        plot_charges(qtopos, acc_charges, tuple([fig, ax]))  # Отрисовка зарядов
        ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


# Расчётная сетка
def create_mesh(x_minmax: tuple, y_minmax: tuple, n_xy: tuple):
    """Равномерно расположить расчётные точки на осях.

    * `x_minmax` и `y_minmax` - границы области
      по соответствующим осям.
    * `n_xy` - число точек по соответствующим осям.

    Возвращает расчётную сетку, созданную как `np.meshgrid`.
    """
    x = np.linspace(*x_minmax, n_xy[0])
    y = np.linspace(*y_minmax, n_xy[1])
    return np.asarray(np.meshgrid(x, y))


def plot_axis(mesh, figax=None):
    """Визуализировать оси.

    * `mesh` - расчётная сетка.
    * `figax` - кортеж вида `(figure, axis)`.

    Возвращает кортеж `(figure, axis)`.
    """
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    x, y = mesh
    ax.plot(x.flatten(), y.flatten(), ls="")
    ax.set(xlabel="$x$, м", ylabel="$y$, м")
    return fig, ax


# Разница потенциалов.
def create_V_field(x, y, qtopos):
    """Рассчитывает поле потенциалов.

    * `x` - координаты расчётной сетки по Ox.
    * `y` - координаты расчётной сетки по Oy.
    * `qtopos` - список зарядов.
    """
    v = []
    for xx, yy in zip(x, y):
        v.append(_V_total(xx, yy, qtopos))
    return np.array(v)


def _V_total(x, y, charges):
    V = 0
    for c in charges:
        Vp = _V_point(c[0], x, y, c[1:])
        V += Vp
    return K * V


def _V_point(q, x, y, a: list):
    return q / ((x - a[0]) ** 2 + (y - a[1]) ** 2) ** 0.5


def plot_potential(list_of_charges: list, mesh, figax: tuple, n: int, freq: float):
    """Отрисовывает эквипотенциальные линии

    * `charges` - список зарядов.
    * `mesh` - расчётная сетка.
    * `figax` - кортеж вида `(fig, ax)`.
    * `n` - точность разбиения линейных зарядов.
    * `freq` - частота эквипотенциальных линий.
    """
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    qtopos = []  # Отформатированный список зарядов
    for i in list_of_charges:
        qtopos += Charge(i).concatenation(n, False)
    qtopos = np.vstack(qtopos)

    x, y = mesh
    v = create_V_field(x, y, qtopos)

    levels = []
    v_sort = sorted(set(v.flatten()))
    for i in range(len(v_sort)):
        if len(levels):
            if i % (len(v_sort) // (10 * freq)) == 0:
                levels.append(v_sort[i])
        else:
            if abs(v_sort[i]) >= 0:
                levels.append(v_sort[i])

    v = v.reshape(len(x), len(y))
    if len(set(levels)) > 1:
        ax.contour(x, y, v, levels=levels, colors=V_COLOR, linestyles='dashed')

    return fig


# Напряжённость
def make_vector_field(list_of_charges, n: int = 1):
    """Возвращает функцию поля `E(mesh)`.

    * `list_of_charges` - список зарядов.
    * `n` - точность разбиения линейных зарядов.
    """
    qtopos = []  # Отформатированный список зарядов
    for i in list_of_charges:
        qtopos += Charge(i).concatenation(n, False)
    qtopos = np.vstack(qtopos)

    qs = qtopos[:, 0]  # Список значений зарядов
    rs = qtopos[:, [1, 2]]  # Список координат зарядов

    def field(mesh):
        """

        * `mesh` - расчётная сетка
        """
        mx, my = mesh
        r = np.vstack([mx.flatten(), my.flatten()])
        return K * _superposition(qs, rs, r).reshape(np.shape(mesh))

    return field


def _superposition(qs, rs, r):
    return np.sum([
                _calc_partial_field(qi, ri, r)
                for qi, ri in zip(qs, rs)
            ], axis=0)


def _calc_partial_field(qi, ri, r):
    dr = r.T - ri
    return qi * dr.T / lina.norm(dr, axis=1) ** 3


def plot_vector_field(mesh, field, freq: float, figax: tuple = None):
    """Визуализировать векторное поле `field`.

    * `mesh` - двумерная расчётная сетка.
    * `freq` - частота отрисовки линий.
    * `field` - двумерное векторное поле.
    * `figax` - кортеж вида `(figure, axis)`.

    Возвращает кортеж вида `(figure, axis)`.
    """
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    mx, my = mesh
    fx, fy = field
    ax.streamplot(mx, my, fx, fy,
                  color=POWER_COLOR, density=freq / 3.33,
                  linewidth=1, arrowstyle='->',
                  arrowsize=1, broken_streamlines=False)

    return fig, ax


# Отрисовка зарядов
def plot_charges(list_of_charges, n: int = 1, figax: tuple = None):
    """Отобразить заряды.

    * `list_of_charges` - список зарядов вида: `[[q, x1, y1, x2, y2], ...]`.
    * `n` - точность разбиения линейных зарядов.
    * `figax` - кортеж вида `(figure, axis)`.

    Возвращает кортеж вида `(figure, axis)`.
    """
    qtopos = []  # Отформатированный список зарядов
    for i in list_of_charges:
        qtopos.append(Charge(i).concatenation(n, True))

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    kw_positive = dict(c="red", ls="", marker="o")
    kw_positive_line = dict(c="red", ls="-", linewidth=4)
    kw_neutral_line = dict(c="lightgrey", ls="-", linewidth=4)
    kw_negative_line = dict(c="dodgerblue", ls="-", linewidth=4)
    kw_neutral = dict(c="lightgrey", ls="", marker="o")
    kw_negative = dict(c="dodgerblue", ls="", marker="o")
    ax.plot([], [], **kw_positive_line)
    ax.plot([], [], **kw_neutral_line)
    ax.plot([], [], **kw_negative_line)
    ax.plot([], [], label="Заряд $+$", **kw_positive)
    ax.plot([], [], label="Заряд $-$", **kw_negative)
    ax.plot([], [], label="Заряд $0$", **kw_neutral)

    for i in qtopos:
        qi = i[0]
        xi = i[1]
        yi = i[2]
        if len(xi) == 1:
            if qi > 0:
                kw = kw_positive
            elif qi == 0:
                kw = kw_neutral
            else:
                kw = kw_negative
        else:
            if qi > 0:
                kw = kw_positive_line
            elif qi == 0:
                kw = kw_neutral_line
            else:
                kw = kw_negative_line
        ax.plot(xi, yi, **kw)

    return fig, ax


def plot_variable_charge(fig: Figure, x: float, y: float, x1: float, y1: float, q: float,
                         eraser: bool = False, light: bool = False):
    """Отрисовывает заряд во время изменения его значения

    * `fig` - фигура графика.
    * `x` - первая координата по Ox.
    * `y` - первая координата по Oy.
    * `x1` - вотрая координата по Ox.
    * `y1` - вотрая координата по Oy.
    * `q` - значение заряда.
    * `eraser` - режим ластика.
    * `light` - режим подсветки.
    """

    ax = fig.get_axes()[0]

    if (x == x1) and (y == y1):
        kw = dict(marker='o')
    else:
        kw = dict(ls="-", linewidth=4)

    if eraser:
        ax.plot([x, x1], [y, y1], color='white', **kw)
        return fig

    if light:
        if q > 0:
            ax.plot([x, x1], [y, y1], color='darkred', **kw)
        elif q < 0:
            ax.plot([x, x1], [y, y1], color='blue', **kw)
        else:
            ax.plot([x, x1], [y, y1], color='grey', **kw)
        return fig

    if q > 0:
        ax.plot([x, x1], [y, y1], color='red', **kw)
    elif q < 0:
        ax.plot([x, x1], [y, y1], color='dodgerblue', **kw)
    else:
        ax.plot([x, x1], [y, y1], color='lightgrey', **kw)
    return fig


# ---------------------------------------------------
class Charge:
    def __init__(self, i):
        self.charges = []
        self.q = i[0]
        self.x1, self.y1 = i[1], i[2]
        self.x2, self.y2 = i[3], i[4]

    def _separation(self, n):
        if (self.x1 == self.x2) and (self.y1 == self.y2):  # Если заряд точечный
            self.x = [self.x1]
            self.y = [self.y1]
        else:  # Если заряд линейный
            self.x = np.linspace(self.x1, self.x2, n)
            self.y = np.linspace(self.y1, self.y2, n)

    def create_charges(self):
        """Создаёт массив зарядов.
        """
        for i in range(len(self.x)):
            self.charges += [[self.q, self.x[i], self.y[i]]]
        return self.charges

    def concatenation(self, n: int, t: bool):
        """Форматирует массив зарядов.

        * `n` - точность разбиения линейных зарядов.
        * `t` - тип работы функции: `False` - построение полей, `True` - построение зарядов.
        """
        self._separation(n)

        if not t:
            return self.create_charges()

        return [((self.q / n) if len(self.x) == 1 else self.q), self.x, self.y]
