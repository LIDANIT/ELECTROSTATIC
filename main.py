import matplotlib.figure
from PyQt6.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton, QTabWidget,
                             QGridLayout, QComboBox, QLineEdit, QLabel, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontMetrics, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from field import *
import sys

text = ('Вводимые параметры:'
        '\n"Xнач." - начальная координата графика по оси абсцисс в метрах. (Примеры: 0; -1.5; 3.0.)'
        '\n"Yкон." - начальная координата графика по оси ординат в метрах. (Примеры: 0; -1.5; 3.0.)'
        '\n"Xнач." - конечная координата графика по оси абсцисс в метрах. (Примеры: 0; -1.5; 3.0.)'
        '\n"Yкон." - конечная координата графика по оси ординат в метрах. (Примеры: 0; -1.5; 3.0.)'
        '\n"Частота линий" - отвечает за частоту отрисовки эквипотенциальных и силовых линий, должно быть'
        ' неотрицательным числом. (Примеры: 0.5; 5.))'
        '\n"Заряд" - заряд в кулонах, который Вы хотите дать телу. (Примеры: -5; 0; 0.5.)'
        '\n"Имя полотна" - название создаваемого полотна для рисования нового графика.'
        ' (Примеры: График №1; Новый график.)'
        '\n"0" (Рядом с кнопкой "Удалить заряд") - заряд выбранного объекта. Ввести число для измерения заряда.'
        ' (Примеры: 0; -5; 3.5.)'
        '\n'
        '\nПереключатели:'
        '\n"Эквипотенциалы" - вкл./выкл. отрисовку эквипотенциальных линий. (Вкл. - голубой, выключено - белый.)'
        '\n"Силовые линии" - вкл./выкл. отрисовку силовых линий. (Вкл. - голубой, выключено - белый.)'
        '\n"Точка" - построить точечный заряд'
        '\n"Прямая" - построить линейный заряд.'
        '\n'
        '\nКнопки:'
        '\n"Построить график" - построить график с выбранными параметрами для поставленных зарядов.'
        '\n"Удалить график" - очистить поле для построения графика.'
        '\n"Новое полотно" - создать полотно для построения графика.'
        '\n"Удалить полотно" - удалить текущее полотно.'
        '\n"Удалить заряд" - удалить выбранное тело.'
        '\n"Помощь" - открыть это окно.'
        '\n"Сохранить график" - сохранить текущее полотно по указанному пути. (По умолчанию - .png.)')


#  Вспомогательные функции.
def validation_float(n: str):
    """

    :param n: проверяемая строка
    :return: True, если можно преобразовать в float
    """
    if n == '':
        return True
    try:
        float(n)
        return True
    except ValueError:
        return False


#  Основная функция окна
def create_window():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowIcon(QIcon('Electrostatic.png'))
    window.show()
    sys.exit(app.exec())


#  Функция создания заряда
def new_charge(tabs: QTabWidget, button: QPushButton, other_button: QPushButton, _type: str):
    """

    :param tabs: вкладка виджетов
    :param button: проверяемая кнопка
    :param other_button: выключаемая кнопка
    :param _type: 'point', если необходимо поставить точку
    :return: создание заряда типа _type
    """
    if tabs.count():
        graph = tabs.currentWidget()
        graph.value = button.isChecked()
        graph._type = _type
        other_button.setChecked(False)


#  Функция отрисовки графика
def _create_graph(settings: QWidget, tabs: QTabWidget):
    """

    :param settings: вкладка с настройками
    :param tabs: вкладка графиков
    :return: отрисовка графика
    """
    if tabs.count():
        plt.delaxes(plt.subplot())
        plt.close()
        graph = tabs.currentWidget()
        x_min = settings.findChild(LineEdit, 'x0')
        y_min = settings.findChild(LineEdit, 'y0')
        x_max = settings.findChild(LineEdit, 'x1')
        y_max = settings.findChild(LineEdit, 'y1')
        freq = settings.findChild(LineEdit, 'freq')

        x_min.setStyleSheet('border: none; margin: 2px')
        y_min.setStyleSheet('border: none; margin: 2px')
        x_max.setStyleSheet('border: none; margin: 2px')
        y_max.setStyleSheet('border: none; margin: 2px')
        freq.setStyleSheet('border: none; margin: 2px')

        validate = 1
        if not validation_float(x_min.text()):
            validate = 0
            x_min.setStyleSheet('border: 2px solid red; margin: 1px')
        if not validation_float(y_min.text()):
            validate = 0
            y_min.setStyleSheet('border: 2px solid red; margin: 1px')
        if not validation_float(x_max.text()):
            validate = 0
            x_max.setStyleSheet('border: 2px solid red; margin: 1px')
        if not validation_float(y_max.text()):
            validate = 0
            y_max.setStyleSheet('border: 2px solid red; margin: 1px')
        if (not validation_float(freq.text())) and ('-' not in freq.text()):
            validate = 0
            freq.setStyleSheet('border: 2px solid red; margin: 1px')

        arrows = settings.findChild(Button, 'arrows').isChecked()
        circle = settings.findChild(Button, 'circle').isChecked()

        if validate:
            graph.x_min = float(x_min.text() or str(graph.x_min))
            graph.y_min = float(y_min.text() or str(graph.y_min))
            graph.x_max = float(x_max.text() or str(graph.x_max))
            graph.y_max = float(y_max.text() or str(graph.y_max))
            graph.freq = float(freq.text() or str(graph.freq))
            if graph.x_min < graph.x_max:
                if graph.y_min < graph.y_max:
                    fig = create_graf(graph.charges, graph.x_min, graph.y_min, graph.x_max, graph.y_max,
                                      graph.acc_power, graph.acc_charges, graph.freq, circle, arrows)
                    graph.canvas_update(fig)
                else:
                    y_min.setStyleSheet('border: 2px solid red; margin: 1px')
                    y_max.setStyleSheet('border: 2px solid red; margin: 1px')
            else:
                x_min.setStyleSheet('border: 2px solid red; margin: 1px')
                x_max.setStyleSheet('border: 2px solid red; margin: 1px')


#  Функция удаления графика
def _del_graph(tabs: QTabWidget):
    """

    :param tabs: вкладка графиков
    :return: стирает текущий график
    """
    if tabs.count():
        graph = tabs.currentWidget()
        graph.charges = []
        tabs.parent().findChild(Container, 'settings').findChild(ComboBox).clear()
        plt.delaxes(plt.subplot())
        plt.close()
        graph.canvas_update(
            create_graf(graph.charges, graph.x_min, graph.y_min, graph.x_max, graph.y_max, graph.acc_power,
                        graph.acc_charges, graph.freq))


#  Функция переключения вкладки
def other_tab(main_container: QWidget):
    """

    :param main_container: главный виджет
    :return: обрабатывает переключение на другую вкладку
    """
    settings = main_container.findChild(Container, 'settings')
    graph = main_container.findChild(QTabWidget).currentWidget()
    settings.findChild(Button, 'new_point_charge').setChecked(False)
    settings.findChild(Button, 'new_line_charge').setChecked(False)
    settings.findChild(ComboBox).update_charges(graph.charges)


#  Функция создания нового полотна для графика
def _new_graph_plot(tabs: QTabWidget, name: str = 'График'):
    """

    :param tabs: вкладка графиков
    :param name: имя полотна
    :return: создаёт новое полотно
    """
    graph = Graph(None)
    tabs.addTab(graph, name)


#  Функция удаления текущего полотна для графика
def _del_graph_plot(tabs: QTabWidget):
    """

    :param tabs: вкладка графиков
    :return: удаляет текущее полотно
    """
    _del_graph(tabs)
    if tabs.count() > 1:
        tabs.removeTab(tabs.currentIndex())
    else:
        _new_graph_plot(tabs)
        tabs.removeTab(tabs.currentIndex())


#  Функция открытия окна помощи
def open_help():
    """

    :return: открытие окна помощи
    """
    window = QMessageBox()
    window.setText(text)
    window.setWindowTitle('Помощь')
    window.setWindowIcon(QIcon('Electrostatic.png'))
    window.show()
    window.exec()


class Font(QFont):
    def __init__(self, size: int = 20):
        """

        :param size: размер шрифта
        """
        super(Font, self).__init__()
        self.setPointSize(size)


class Container(QWidget):
    def __init__(self, parent: QWidget, name: str = None):
        """

        :param parent: виджет-родитель
        :param name: имя виджета
        """
        super(Container, self).__init__()
        self.setObjectName(name)
        self.setParent(parent)


class Label(QLabel):
    def __init__(self, text_of_label: str, name: str = None, font: QFont = Font()):
        """

        :param text_of_label: текст
        :param name: имя виджета
        :param font: шрифт
        """
        super(Label, self).__init__(text_of_label)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName(name)
        self.setFont(font)
        self.font_metrics = QFontMetrics(font)
        self.setFixedHeight(self.font_metrics.height() * 3)
        self.setAlignment(Qt.AlignmentFlag.AlignTop)


class Button(QPushButton):
    def __init__(self, text_of_button: str, _type: bool = False, name: str = None):
        """

        :param text_of_button: текст кнопки
        :param _type: тип кнопки(1 - рычаг, 0 - кнопка)
        :param name: имя виджета кнопки
        """
        super(Button, self).__init__(text_of_button)
        self.setObjectName(name)
        self.setFont(Font())
        self.setCheckable(_type)


class LineEdit(QLineEdit):
    def __init__(self, text_of_label: str, name: str = None):
        """

        :param text_of_label: текст заднего фона
        :param name: имя виджета строки ввода
        """
        super(LineEdit, self).__init__()
        self.setPlaceholderText(text_of_label)
        self.setObjectName(name)
        self.setFont(Font())


class ComboBox(QComboBox):
    def __init__(self, parent: QWidget, font: QFont):
        """

        :param parent: виджет-родитель
        :param font: шрифт
        """
        super(ComboBox, self).__init__(parent)
        self.setFont(font)
        self.charges = []
        self.currentIndexChanged.connect(self.show_charge)
        self.lastIndex = None

    def update_charges(self, charges: list):
        """

        :param charges: список зарядов
        :return: обновляет комбо-бокс
        """
        self.charges = charges
        self.clear()
        for i in self.charges:
            if (i[1] == i[3]) and (i[2] == i[4]):
                self.addItem(f'Заряд точечный, {i[0]}Кл.')
            else:
                self.addItem(f'Заряд линейный, {i[0]}Кл.')

    def show_charge(self):
        if self.count():
            text_of_combo = self.currentText()[16:-3]
            self.parent().findChild(LineEdit, 're_charge').setText(text_of_combo)
            self.light()
            self.lastIndex = self.currentIndex()

    def light(self):
        graph = self.parent().parent().findChild(QTabWidget).currentWidget()
        if not (self.lastIndex is None):
            graph.charge_light_off(self.lastIndex)
        graph.charge_light_on(self.currentIndex())

    def recharge(self, entry: QLineEdit):
        """

        :param entry: строка ввода
        :return: отображает текущий заряд в панели обновления его значения
        """
        graph = self.parent().parent().findChild(QTabWidget).currentWidget()
        if validation_float(entry.text()):
            entry.setStyleSheet('border: none; margin: 2px')
            if self.count():
                text_of_entry = entry.text() or '0'
                if text_of_entry == '-':
                    text_of_entry = '0'
                self.setItemText(self.currentIndex(), self.currentText()[:16] + str(float(text_of_entry)) + 'Кл.')
                graph.recharge(text_of_entry, self.currentIndex())
        else:
            entry.setStyleSheet('border: 2px solid red; margin: 1px')

    def delete_charge(self):
        if self.count():
            graph = self.parent().parent().findChild(QTabWidget).currentWidget()
            self.removeItem(self.currentIndex())
            graph.delete_charge(self.currentIndex())


class Graph(Container):
    def __init__(self, parent: QWidget = None, name: str = None):
        """

        :param parent: виджет-родитель
        :param name: имя виджета
        """
        super().__init__(parent)
        self.save_btn = None
        self.setObjectName(name)
        self.layout_graph = QGridLayout(self)

        self.value = 0
        self.q = 0
        self.charges = []
        self.x_min = 0
        self.y_min = 0
        self.x_max = 10
        self.y_max = 10
        self.freq = 2
        self.acc_power = 100
        self.acc_charges = 100

        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        self.canvas = None
        self._type = 'point'
        self.file = QFileDialog()

        fig = create_graf(self.charges, self.x_min, self.y_min, self.x_max, self.y_max, self.acc_power,
                          self.acc_charges, self.freq)
        self.canvas_update(fig)
        self.setLayout(self.layout_graph)

    def get_cords(self, event):
        """
        :return: получает координаты курсора с графика
        """
        mx, my = event.x, event.y
        x, y = self.canvas.figure.get_axes()[0].transData.inverted().transform([mx, my])
        if self.value and \
                (self.canvas.figure.get_axes()[0].get_xlim()[1] > x > self.canvas.figure.get_axes()[0].get_xlim()[
                    0]) and \
                (self.canvas.figure.get_axes()[0].get_ylim()[1] > y > self.canvas.figure.get_axes()[0].get_ylim()[0]):
            self.add_to_stack(x, y)
        else:
            self.add_to_stack(None, None)

    def add_to_stack(self, x: int, y: int):
        """

        :param x: координата x
        :param y: координата y
        :return: добавляет выбранные координаты в очередь
        """
        if self.x0 is None:
            self.x0 = x
            self.y0 = y
            if self._type == 'point' and not (self.x0 is None):
                self.add_charge(self.x0, self.y0, self.x0, self.y0)
                self.x0 = None
                self.y0 = None
        else:
            self.x1 = x
            self.y1 = y
            if not (self.x1 is None):
                self.add_charge(self.x0, self.y0, self.x1, self.y1)
                self.x0 = None
                self.y0 = None
                self.x1 = None
                self.y1 = None

    def add_charge(self, x0, y0, x1, y1):
        """
        :return: добавляет заряд в отрисовку и к графику
        """
        if not ((x0 is None) or (y0 is None)):
            if validation_float(self.parent().parent().parent().findChild(Container, 'settings').findChild(
                    LineEdit, 'charge_value').text()):
                self.parent().parent().parent().findChild(Container, 'settings').findChild(
                    LineEdit, 'charge_value').setStyleSheet(
                    'border: none; margin: 2px')
                self.q = float(self.parent().parent().parent().findChild(Container, 'settings').findChild(
                    LineEdit, 'charge_value').text() or '0')
                self.canvas_update(plot_variable_charge(self.canvas.figure, x0, y0, x1, y1, self.q))
                self.charges.append([self.q, x0, y0, x1, y1])
                for i in self.charges:
                    if not len(i):
                        del i
                self.parent().parent().parent().findChild(Container, 'settings').findChild(ComboBox).update_charges(
                    self.charges)
            else:
                self.parent().parent().parent().findChild(Container, 'settings').findChild(
                    LineEdit, 'charge_value').setStyleSheet(
                    'border: 2px solid red; margin: 1px')

    def canvas_update(self, fig: matplotlib.figure.Figure):
        """
        :param fig: фигура
        :return: обновление канваса
        """
        for i in self.findChildren(Canvas) + self.findChildren(Button):
            i.deleteLater()
        self.canvas = Canvas(fig)
        self.canvas.mpl_connect('button_press_event', self.get_cords)
        self.save_btn = Button('Сохранить график', name='save')
        self.save_btn.clicked.connect(self.open_file)
        self.layout_graph.addWidget(self.canvas, 0, 0)
        self.layout_graph.addWidget(self.save_btn)
        self.canvas.draw()

    def open_file(self):
        self.file.setLabelText(self.file.DialogLabel.Accept, '123')
        text_btn = self.file.getSaveFileName(caption='Сохранить график')[0]
        self.canvas.figure.savefig(text_btn) if text_btn else 0

    def recharge(self, text_of_charge: str, index: int):
        """

        :param text_of_charge: значение заряда
        :param index: индекс заряда
        :return: изменяет значение заряда
        """
        self.charges[index][0] = float(text_of_charge)
        self.canvas_update(plot_variable_charge(self.canvas.figure, self.charges[index][1], self.charges[index][2],
                                                self.charges[index][3], self.charges[index][4], self.charges[index][0]))

    def delete_charge(self, i: int):
        """

        :param i: индекс заряда
        :return: удаляет выбранный заряд
        """
        self.canvas_update(plot_variable_charge(self.canvas.figure, self.charges[i][1], self.charges[i][2],
                                                self.charges[i][3], self.charges[i][4], self.charges[i][0],
                                                eraser=True))
        del self.charges[i]

    def charge_light_on(self, i: int):
        """

        :param i: индекс заряда
        :return: подсвечивает выбранный заряд
        """
        self.canvas_update(plot_variable_charge(self.canvas.figure, self.charges[i][1], self.charges[i][2],
                                                self.charges[i][3], self.charges[i][4], self.charges[i][0],
                                                light=True))

    def charge_light_off(self, i: int):
        """

        :param i: индекс заряда
        :return: убирает подсветку заряда
        """
        self.canvas_update(plot_variable_charge(self.canvas.figure, self.charges[i][1], self.charges[i][2],
                                                self.charges[i][3], self.charges[i][4], self.charges[i][0]))


class Canvas(FigureCanvasQTAgg):
    def __init__(self, fig: matplotlib.figure.Figure):
        """

        :param fig: фигура графика
        """
        super(Canvas, self).__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Электростатика')
        self.setMinimumSize(1600, 800)
        main_container = Container(self)

        layout = QGridLayout(main_container)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 1)

        settings = Container(main_container, 'settings')
        graph_tabs = QTabWidget(main_container)
        graph_tabs.setFont(Font(15))
        start_graph = Graph(None)
        graph_tabs.addTab(start_graph, 'График')

        # tools
        layout_settings = QGridLayout(settings)
        layout_settings.setColumnStretch(0, 1)
        layout_settings.setColumnStretch(1, 200)
        layout_settings.setVerticalSpacing(0)
        layout_settings.setHorizontalSpacing(0)

        cords_cont = Label('Начальные и конечные \nкоординаты графика', 'cords_cont')
        cords_cont.setFont(Font())
        cords_cont.setMaximumHeight(40)
        x0 = LineEdit('Xнач.', 'x0')
        y0 = LineEdit('Yнач.', 'y0')
        x1 = LineEdit('Xкон.', 'x1')
        y1 = LineEdit('Yкон.', 'y1')
        freq = LineEdit('Частота линий', 'freq')

        x0.setStyleSheet('border: none; margin: 2px')
        y0.setStyleSheet('border: none; margin: 2px')
        x1.setStyleSheet('border: none; margin: 2px')
        y1.setStyleSheet('border: none; margin: 2px')
        freq.setStyleSheet('border: none; margin: 2px')

        circle_btn = Button('Эквипотенциалы', True, name='circle')
        circle_btn.setChecked(True)
        arrows_btn = Button('Силовые линии', True, name='arrows')
        arrows_btn.setChecked(True)

        charge_value = LineEdit('Заряд', 'charge_value')
        charge_value.setStyleSheet('border: none; margin: 2px')
        new_point_charge = Button('Точка', _type=True, name='new_point_charge')
        new_point_charge.clicked.connect(lambda: new_charge(graph_tabs, new_point_charge, new_line_charge, 'point'))
        new_line_charge = Button('Прямая', _type=True, name='new_line_charge')
        new_line_charge.clicked.connect(lambda: new_charge(graph_tabs, new_line_charge, new_point_charge, 'line'))
        graph_tabs.currentChanged.connect(lambda: other_tab(main_container))

        create_graph = Button('Построить график')
        create_graph.clicked.connect(lambda: _create_graph(settings, graph_tabs))
        del_graph = Button('Удалить график')
        del_graph.clicked.connect(lambda: _del_graph(graph_tabs))
        new_graph_plot = Button('Новое полотно')
        graph_plot_name = LineEdit('Имя полотна', 'qraph_plot_name')
        new_graph_plot.clicked.connect(lambda: _new_graph_plot(graph_tabs, name=graph_plot_name.text() or 'График'))
        del_graph_plot = Button('Удалить полотно')
        del_graph_plot.clicked.connect(lambda: _del_graph_plot(graph_tabs))

        list_of_charges = ComboBox(settings, Font())
        delete_charge = Button('Удалить заряд')
        re_charge = LineEdit('0', 're_charge')
        re_charge.setStyleSheet('border: none; margin: 2px')
        re_charge.textEdited.connect(lambda: list_of_charges.recharge(re_charge))
        delete_charge.clicked.connect(list_of_charges.delete_charge)
        help_btn = Button('Помощь', name='help')
        help_btn.clicked.connect(open_help)

        # layouts settings
        layout.addWidget(graph_tabs, 0, 0)
        layout.addWidget(settings, 0, 1)

        layout_settings.addWidget(cords_cont, 0, 0, 1, 0)
        layout_settings.addWidget(x0, 1, 0)
        layout_settings.addWidget(y0, 1, 1)
        layout_settings.addWidget(x1, 2, 0)
        layout_settings.addWidget(y1, 2, 1)
        layout_settings.addWidget(freq, 3, 0, 1, 0)
        layout_settings.addWidget(circle_btn, 4, 0)
        layout_settings.addWidget(arrows_btn, 4, 1)
        layout_settings.addWidget(new_point_charge, 5, 0)
        layout_settings.addWidget(new_line_charge, 5, 1)
        layout_settings.addWidget(charge_value, 6, 0, 1, 0)
        layout_settings.addWidget(create_graph, 7, 0, 1, 0)
        layout_settings.addWidget(del_graph, 8, 0, 1, 0)
        layout_settings.addWidget(new_graph_plot, 9, 0)
        layout_settings.addWidget(graph_plot_name, 9, 1)
        layout_settings.addWidget(del_graph_plot, 10, 0, 1, 0)
        layout_settings.addWidget(list_of_charges, 11, 0, 1, 0)
        layout_settings.addWidget(delete_charge, 12, 0)
        layout_settings.addWidget(re_charge, 12, 1)
        layout_settings.addWidget(help_btn, 13, 0, 1, 0)

        for row in range(layout_settings.rowCount()):
            layout_settings.setRowStretch(row, 1)
        for col in range(layout_settings.columnCount()):
            layout_settings.setColumnStretch(col, 1)

        main_container.setLayout(layout)
        settings.setLayout(layout_settings)
        self.setCentralWidget(main_container)


if __name__ == '__main__':
    create_window()
