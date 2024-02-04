import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lina
from scipy.constants import pi, epsilon_0

K = 1 / (4 * pi * epsilon_0)
POWER_COLOR = "black"
V_COLOR = "black"

def plot_variable_charge(fig, x, y, x1, y1, q, eraser=0, light=0):
    ax = fig.get_axes()[0]
    if (x == x1) and (y == y1):
        kw = dict(marker='o')
    else:
        kw = dict(ls="-", linewidth=4)
    if eraser:
        ax.plot([x, x1], [y, y1], color='white', **kw)
    else:
        if light:
            if q > 0:
                ax.plot([x, x1], [y, y1], color='darkred', **kw)
            elif q < 0:
                ax.plot([x, x1], [y, y1], color='blue', **kw)
            else:
                ax.plot([x, x1], [y, y1], color='grey', **kw)
        else:
            if q > 0:
                ax.plot([x, x1], [y, y1], color='red', **kw)
            elif q < 0:
                ax.plot([x, x1], [y, y1], color='dodgerblue', **kw)
            else:
                ax.plot([x, x1], [y, y1], color='lightgrey', **kw)
    return fig


def create_graf(q_arr=None, x0=0, y0=0, x1=10, y1=10, acc_power=100, acc_charges=100, freq=2.0, circle=0, arrows=0):
    if q_arr is None:
        q_arr = []
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    mesh = create_mesh(tuple([x0, x1]), tuple([y0, y1]), tuple([acc_power, acc_power]))
    fig, ax = plot_mesh_2d(mesh, figax=tuple([fig, ax]))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if len(q_arr):
        qtopos = np.vstack(q_arr)
        E = make_vector_field(qtopos, acc_charges)

        if circle:
            plot_potential(q_arr, mesh, tuple([fig, ax]), acc_charges, freq)
        if arrows:
            plot_vector_field_2d(mesh, E(mesh), freq, tuple([fig, ax]))  # Отрисовка поля
        plot_charges(qtopos, acc_charges, tuple([fig, ax]))  # Отрисовка зарядов
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def make_vector_field(list_of_charges, n=1):
    """Возвращает функцию поля `E(mesh)`.

    * `qtopos` - массив размера `n` на `3`, где в
      каждая из `n` строк имеет вид
      `(qi, xi, yi)`.
    """
    qtopos = []
    for i in list_of_charges:
        qtopos += Charge(i).concatenation(n, 0)
    qtopos = np.vstack(qtopos)
    qs = qtopos[:, 0]
    rs = qtopos[:, [1, 2]]

    def field(mesh):
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


def plot_mesh_2d(mesh, figax=None, **kw):
    """Визуализировать двумерную расчётную сетку.

    * `mesh` - расчётная сетка.
    * `figax` - кортеж вида `(figure, axis)`.
      По умолчанию `None`.

    Возвращает кортеж `(figure, axis)`.
    """
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    x, y = mesh
    ax.plot(x.flatten(), y.flatten(), ls="", **kw)
    ax.set(xlabel="$x$, м", ylabel="$y$, м")
    return fig, ax


def plot_vector_field_2d(mesh, field, freq,
                         figax=None, **kw):
    """Визуализировать векторное поле `field`.

    * `mesh` - двумерная расчётная сетка.
    * `field` - двумерное векторное поле.
    * `figax` - кортеж вида `(figure, axis)`.
    * `cbar_label` - название цветовой шкалы, если она есть.

    Возвращает кортеж вида `(figure, axis)`.
    """
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    mx, my = mesh
    fx, fy = field
    ax.streamplot(mx, my, fx, fy, **kw, color=POWER_COLOR, density=freq,
                   linewidth=1, arrowstyle='->',
                   arrowsize=1, broken_streamlines=False)
    return fig, ax


# Дополнительно опишем функцию отрисовки зарядов
def plot_charges(list_of_charges, n, figax=None):
    """Отобразить заряды.

    * `qtopos` - словарь, ключом которого является
      величина заряда `qi`, а значением -
      координата заряда `(xi, yi)`.
    * `figax` - кортеж вида `(figure, axis)`.

    Возвращает кортеж вида `(figure, axis)`.
    """
    qtopos = []
    for i in list_of_charges:
        qtopos.append(Charge(i).concatenation(n, 1))

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


def create_V_field(x, y, qtopos):
    v = []
    for xx, yy in zip(x, y):
        v.append(V_total(xx, yy, qtopos))
    return np.array(v)


def V_total(x, y, charges):
    V = 0
    for c in charges:
        Vp = V_point(c[0], x, y, c[1:])
        V += Vp
    return K * V


def V_point(q, x, y, a: list):
    return q / ((x-a[0])**2 + (y-a[1])**2)**0.5


def plot_potential(charges, mesh, figax, n, freq):
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    qtopos = []
    for i in charges:
        qtopos += Charge(i).concatenation(n, 0)
    qtopos = np.vstack(qtopos)

    x, y = mesh
    v = create_V_field(x, y, qtopos)
    levels = []
    v_sort = sorted(set(v.flatten()))
    for i in range(len(v_sort)):
        if len(levels):
            if i % (len(v_sort)//(10*freq)) == 0:
                levels.append(v_sort[i])
        else:
            if abs(v_sort[i]) >= 0:
                levels.append(v_sort[i])
    v = v.reshape(len(x), len(y))
    if len(set(levels)) > 1:
        ax.contour(x, y, v, levels=levels, colors=V_COLOR, linestyles='dashed')
    return fig


# ---------------------------------------------------


class Charge:
    def __init__(self, i):
        self.charges = []
        self.q = i[0]
        self.x1, self.y1 = i[1], i[2]
        self.x2, self.y2 = i[3], i[4]

    def _separation(self, n):
        if (self.x1 == self.x2) and (self.y1 == self.y2):
            self.x = [self.x1]
            self.y = [self.y1]
        else:
            self.x = np.linspace(self.x1, self.x2, n)
            self.y = np.linspace(self.y1, self.y2, n)

    def create_charges(self):
        for i in range(len(self.x)):
            self.charges += [[self.q, self.x[i], self.y[i]]]
        return self.charges

    def concatenation(self, n, t):
        self._separation(n)
        if t == 0:
            return self.create_charges()
        elif t == 1:
            return [((self.q / n) if len(self.x) == 1 else self.q), self.x, self.y]
