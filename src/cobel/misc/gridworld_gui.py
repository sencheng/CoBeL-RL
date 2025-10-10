# basic imports
import sys
import math
import pickle
import numpy as np
# Qt imports
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QPen,
    QBrush,
    QColor,
    QColorConstants,
    QIntValidator,
    QAction,
    QCloseEvent,
    QWheelEvent,
    QShowEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QDialog,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QRadioButton,
    QTableWidget,
    QGraphicsScene,
    QLineEdit,
    QGraphicsView,
    QSplitter,
    QPushButton,
    QFileDialog,
    QDialogButtonBox,
    QMessageBox,
    QComboBox,
    QMenuBar,
    QMenu,
    QGraphicsTextItem,
    QGraphicsSceneMouseEvent,
    QGraphicsLineItem,
)
from PyQt6 import QtWidgets
# framework imports
import cobel.misc.gridworld_tools as gwt
import cobel.misc.gridworld_export as gwe
# typing
from numpy.typing import NDArray
from ..interface.gridworld import WorldDict


class MainWindow(QMainWindow):
    """
    The gridworld editor's main window class.

    Parameters
    ----------
    world : dict or None, optional
        The gridworld dictionary (5 x 5 open field by default).

    """

    def __init__(self, world: None | WorldDict = None, *args, **kwargs) -> None:
        """
        The gridworld editor's main window class.

        Parameters
        ----------
        world : dict or None, optional
            The gridworld dictionary (5 x 5 open field by default).
        """
        super().__init__(*args, **kwargs)
        # gridworld
        self.world = gwt.make_open_field(5, 5) if world is None else world
        self.changed = False
        self.file: str | None = None
        self.resize(50, 50)
        self.border = 5
        self.index = 0
        # init GUI
        self.setWindowTitle('Gridworld Editor')
        self.create_MenuBar()
        self.central_panel = CentralPanel(self)
        self.setCentralWidget(self.central_panel)
        # window size
        self.resize(1000, 563)

    def create_MenuBar(self) -> None:
        """
        This function creates the menu bar.
        """
        # Define actions
        # new
        self.action_new = QAction('&New', self)
        self.action_new.setShortcut('Ctrl+N')
        self.action_new.triggered.connect(self.new)
        # open
        self.action_open = QAction('&Open', self)
        self.action_open.setShortcut('Ctrl+O')
        self.action_open.triggered.connect(self.load)
        # save
        self.action_save = QAction('&Save', self)
        self.action_save.setShortcut('Ctrl+S')
        self.action_save.triggered.connect(self.save)
        # save as
        self.action_save_as = QAction('&Save as', self)
        self.action_save_as.setShortcut('Ctrl+Shift+S')
        self.action_save_as.triggered.connect(self.save_as)
        # quit
        self.action_quit = QAction('&Quit', self)
        self.action_quit.setShortcut('Ctrl+Q')
        self.action_quit.triggered.connect(self.close)
        # export
        self.action_export = QAction('&Export')
        self.action_export.setShortcut('Ctrl+E')
        self.action_export.triggered.connect(self.export)
        # T-maze
        self.action_tmaze = QAction('&T-Maze', self)
        self.action_tmaze.triggered.connect(self.open_tmaze_dialog)
        # 8-maze
        self.action_eightmaze = QAction('&8-Maze', self)
        self.action_eightmaze.triggered.connect(self.open_eightmaze_dialog)
        # Two-choice T-maze
        self.action_twochoice_tmaze = QAction('&Two choice T-Maze', self)
        self.action_twochoice_tmaze.triggered.connect(self.open_twochoice_dialog)
        # Detour Maze
        self.action_detour = QAction('&Detour Maze', self)
        self.action_detour.triggered.connect(self.open_detour_dialog)
        # Two-sided T-maze
        self.action_twosided_tmaze = QAction('&Two-sided T-Maze', self)
        self.action_twosided_tmaze.triggered.connect(self.open_twosided_tmaze_dialog)
        # Double T-maze
        self.action_double_tmaze = QAction('&Double T-Maze', self)
        self.action_double_tmaze.triggered.connect(self.open_double_tmaze_dialog)
        # info
        self.action_info = QAction('&Info', self)
        self.action_info.setShortcut('Ctrl+I')
        self.action_info.triggered.connect(self.info)
        # create menu nar
        self.menu_bar = self.menuBar()
        assert type(self.menu_bar) is QMenuBar
        # add menus to menu bar
        self.file_menu = self.menu_bar.addMenu('&File')
        self.info_menu = self.menu_bar.addMenu('&Info')
        self.tools_menu = self.menu_bar.addMenu('&Templates')
        assert type(self.file_menu) is QMenu
        assert type(self.info_menu) is QMenu
        assert type(self.tools_menu) is QMenu
        # add actions to menu
        self.file_menu.addAction(self.action_new)
        self.file_menu.addAction(self.action_open)
        self.file_menu.addAction(self.action_save)
        self.file_menu.addAction(self.action_save_as)
        self.file_menu.addAction(self.action_export)
        self.file_menu.addAction(self.action_quit)
        self.info_menu.addAction(self.action_info)
        self.tools_menu.addAction(self.action_tmaze)
        self.tools_menu.addAction(self.action_eightmaze)
        self.tools_menu.addAction(self.action_twochoice_tmaze)
        self.tools_menu.addAction(self.action_detour)
        self.tools_menu.addAction(self.action_twosided_tmaze)
        self.tools_menu.addAction(self.action_double_tmaze)

    def open_tmaze_dialog(self) -> None:
        """
        This function opens the T-maze template dialog.
        """
        self.temps_open = TemplateDialog(self, 't_maze')
        self.temps_open.show()

    def open_eightmaze_dialog(self) -> None:
        """
        This function opens the 8-maze template dialog.
        """
        self.temps_open = TemplateDialog(self, '8_maze')
        self.temps_open.show()

    def open_twochoice_dialog(self) -> None:
        """
        This function opens the Two-choice T-maze template dialog.
        """
        self.temps_open = TemplateDialog(self, 'two_choice_t_maze')
        self.temps_open.show()

    def open_detour_dialog(self) -> None:
        """
        This function opens the detour maze template dialog.
        """
        self.temps_open = TemplateDialog(self, 'detour_maze')
        self.temps_open.show()

    def open_twosided_tmaze_dialog(self) -> None:
        """
        This function opens the Two-sided T-maze template dialog.
        """
        self.temps_open = TemplateDialog(self, 'two_sided_t_maze')
        self.temps_open.show()

    def open_double_tmaze_dialog(self) -> None:
        """
        This function opens the Double T-maze template dialog.
        """
        self.temps_open = TemplateDialog(self, 'double_t_maze')
        self.temps_open.show()

    def new(self) -> None:
        """
        This function opens the new gridworld dialog.
        """
        if not self.unsaved_changes():
            self.new_open = NewDialog(self)
            self.new_open.show()

    def load(self) -> None:
        """
        This function asks the user to select a file and opens it.
        """
        if not self.unsaved_changes():
            file_name = QFileDialog.getOpenFileName(
                self, 'Open file', 'c:\\', 'Pickle files (*.pkl)'
            )[0]
            print('Try loading file: ', file_name)
            try:
                self.world = pickle.load(open(file_name, 'rb'))
                self.changed = False
                self.file = None
                self.index = 0
                self.central_panel = CentralPanel(self)
                self.setCentralWidget(self.central_panel)
                # dirty solution as the previous currently breaks the editor
                world = pickle.load(open(file_name, 'rb'))
                main = MainWindow(world)
                main.show()
                self.close()
                print('Successfully loaded file. ')
            except:
                print("Couldn't load file.")

    def save_as(self) -> None:
        """
        This function saves the current gridworld in a file.
        """
        file_name = QFileDialog.getSaveFileName(
            self, 'Open file', 'c:\\', 'Pickle files (*.pkl)'
        )[0]
        print('Try saving file: ', file_name)
        try:
            pickle.dump(self.world, open(file_name, 'wb'))
            self.changed = False
            self.file = file_name
            print('Successfully saved file. ')
        except:
            print("Couldn't save file.")

    def save(self) -> None:
        """
        This function saves the current gridworld.
        """
        if self.file is None:
            self.save_as()
        else:
            print('Try saving file: ', self.file)
            try:
                pickle.dump(self.world, open(self.file, 'wb'))
                self.changed = False
                print('Successfully saved file. ')
            except:
                print("Couldn't save file.")

    def export(self) -> None:
        """
        This function exports the current gridworld into a Wavefront obj file.
        """
        file_name = QFileDialog.getSaveFileName(
            self, 'export as 3D-Model', '', 'Obj files (*.obj)'
        )[0]
        if file_name:
            invalid_transitions = self.world['invalid_transitions']
            width, height = self.world['width'], self.world['height']
            wall_info, pillars = gwe.retrieve_wall_info(
                invalid_transitions, width, height, state_size=1
            )
            walls = gwe.generate_walls(
                wall_info,
                pillars,
                width,
                height,
                state_size=1,
                wall_height=1,
                wall_depth=0.2,
            )
            gwe.export_as_obj(walls, width, height, state_size=1, file_name=file_name)
        else:
            print('filename not found')

    def unsaved_changes(self) -> bool:
        """
        This function checks for unsaved changes and informs the user accordingly.

        Returns
        -------
        flag : bool
            True, if unsaved changes were found.
        """
        # Returns False if there are no unsaved changes/if these should be ignored
        if self.changed:
            print('Unsaved changes.')
            self.unsaved_changes_dialog = UnsavedChangesDialog(self)
            if self.unsaved_changes_dialog.exec():
                return False
            else:
                return True

        return False

    def info(self) -> None:
        """
        This functions opens the info dialog.
        """
        self.info_dialog = InfoDialog(self)
        self.info_dialog.exec()

    def closeEvent(self, event: None | QCloseEvent) -> None:
        """
        This function closes the program.

        Parameters
        ----------
        event : QCloseEvent or None
            The close event.
        """
        if not self.unsaved_changes():
            super().closeEvent(event)
        else:
            assert event is not None
            event.ignore()


class CentralPanel(QWidget):
    """
    The widget that handles the two main widgets (grid and state information).

    Parameters
    ----------
    parent : MainWindow
        The central panel's parent window (i.e., the main window).

    """

    def __init__(self, parent: MainWindow) -> None:
        super().__init__()
        self.setParent(parent)
        # Grid and View
        self.scene = Grid(self, parent)
        self.view = GridViewer(self.scene, self, parent)
        self.scene.highlight_terminal_starting(
            parent.world['terminals'],
            parent.world['starting_states'],
            parent.world['goals'],
        )
        # Sidebar
        self.state_menu = StateInformation(self, parent)
        # splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.state_menu)  # sidebar here
        self.splitter.setStretchFactor(0, 10)
        # add splitter(Grid & InfoBar) to layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)

    def change_state_information(self, textCoord: str, index: int) -> None:
        """
        This function updates the state information panel.

        Parameters
        ----------
        textCoord : str
            The state's coordinates.
        index : int
            The state's index.
        """
        self.state_menu.change_state_information(textCoord, index)

    def highlight_terminal_starting(
        self, terminals: NDArray, startings: NDArray, goals: list[int]
    ) -> None:
        """
        This function marks states according to whether
        they are terminal or starting states.

        Parameters
        ----------
        terminal : NDArray
            A numpy array containg the gridworld's terminal state's indeces.
        startings : list of int
            A list containing the gridworld's starting state's indeces.
        """
        self.scene.highlight_terminal_starting(terminals, startings, goals)


class Grid(QGraphicsScene):
    """
    This class presents the grid of the gridworld.

    Parameters
    ----------
    parent : CentralPanel
        The grid's parent window.
    main_window : MainWindow
        The gridworld editor's main window.

    """

    def __init__(self, parent: CentralPanel, main_window: MainWindow) -> None:
        super().__init__(parent)
        self.setParent(parent)
        self.main_window = main_window
        self.lines: list[QGraphicsLineItem] = []
        self.highlight = self.addRect(
            self.main_window.border / 2,
            self.main_window.border / 2,
            self.main_window.width() - self.main_window.border,
            self.main_window.height() - self.main_window.border,
            QPen(QColor(125, 175, 240, 0), 0),
            QBrush(QColor(125, 175, 240, 0)),
        )
        self.scene_height = self.main_window.world['height'] * self.main_window.height()
        self.scene_width = self.main_window.world['width'] * self.main_window.width()
        self.pen = QPen(QColor(125, 175, 240, 125), 0)
        self.brush = QBrush(QColor(125, 175, 240, 125))
        self.draw_grid()
        self.symbols_terminal: list[QGraphicsTextItem] = []
        self.symbols_starting: list[QGraphicsTextItem] = []

    def draw_grid(self) -> None:
        """
        This function draws the grid.
        """
        pen_lines = QPen(QColorConstants.Gray, self.main_window.border)
        pen_border = QPen(QColorConstants.Black, self.main_window.border)
        # lines parallel to GUI x-axis (vertical)
        for column in range(1, self.main_window.world['width']):
            x = column * self.main_window.width()
            for row in range(0, self.main_window.world['height']):
                y = row * self.main_window.height()
                line = Line(
                    x, y, x, y + self.main_window.height(), pen_lines, self.main_window
                )
                if line.transition in self.main_window.world['invalid_transitions']:
                    line.highlighted = True
                    line.setPen(QPen(QColorConstants.Red, self.main_window.border))
                self.lines.append(line)
                self.addItem(line)
        # lines parallel to GUI y-axis (horizontal)
        for row in range(1, self.main_window.world['height']):
            y = row * self.main_window.height()
            for column in range(0, self.main_window.world['width']):
                x = column * self.main_window.width()
                line = Line(
                    x, y, x + self.main_window.width(), y, pen_lines, self.main_window
                )
                if line.transition in self.main_window.world['invalid_transitions']:
                    line.highlighted = True
                    line.setPen(QPen(QColorConstants.Red, self.main_window.border))
                self.lines.append(line)
                self.addItem(line)
        # Outer border
        outer_lines = [
            self.addLine(0, 0, self.scene_width, 0, pen_border),
            self.addLine(0, 0, 0, self.scene_height, pen_border),
            self.addLine(
                0, self.scene_height, self.scene_width, self.scene_height, pen_border
            ),
            self.addLine(
                self.scene_width, 0, self.scene_width, self.scene_height, pen_border
            ),
        ]
        for outer_line in outer_lines:
            assert type(outer_line) is QGraphicsLineItem
            self.lines.append(outer_line)

    def set_visible(self, visible: bool = True) -> None:
        """
        This function sets the visibility of the grid.

        Parameters
        ----------
        visible : bool, default=True
            Flag determining the visibility of the grid.
        """
        for line in self.lines:
            line.setVisible(visible)

    def delete_grid(self) -> None:
        """
        This function deletes the grid.
        """
        for line in self.lines:
            self.removeItem(line)
        del self.lines[:]

    def set_opacity(self, opacity: float) -> None:
        """
        This function sets the opacity of the grid lines.

        Parameters
        ----------
        opacity : float
            The grid line opacity.
        """
        for line in self.lines:
            line.setOpacity(opacity)

    def highlight_state(self, x: int, y: int) -> None:
        """
        This function highlights a state.

        Parameters
        ----------
        x : int
            The state's x coordinate.
        y : int
            The state's y coordinate.
        """
        # removes pieces old highlight
        self.removeItem(self.highlight)
        w, h = self.main_window.world['width'], self.main_window.world['height']
        self.highlight = self.addRect(
            y * self.scene_width / w + self.main_window.border / 2,
            x * self.scene_height / h + self.main_window.border / 2,
            self.scene_width / w - self.main_window.border,
            self.scene_height / h - self.main_window.border,
            self.pen,
            self.brush,
        )
        # removes artefacts of old highlights
        self.update()

    def highlight_terminal_starting(
        self, terminals: NDArray, startings: NDArray, goals: list[int]
    ) -> None:
        """
        This function marks states according to whether
        they are terminal or starting states.

        Parameters
        ----------
        terminals : NDArray
            A numpy array containing the terminal states' indeces.
        goals : list of int
            A list containing the goal states' indeces.
        startings : list of int
            A list containing the starting states' indeces.
        """
        # delete all symbols created before and then empty the array
        for _ in range(len(self.symbols_terminal)):
            self.removeItem(self.symbols_terminal[-1])
            self.symbols_terminal.pop(-1)
        for _ in range(len(self.symbols_starting)):
            self.removeItem(self.symbols_starting[-1])
            self.symbols_starting.pop(-1)
        # create symbol at position x,y for all terminals and startings
        w, h = self.main_window.world['width'], self.main_window.world['height']
        for index in np.arange(terminals.shape[0])[terminals == 1]:
            # calculate position
            y = (index // w) * self.scene_width / w
            x = (index % w) * self.scene_height / h
            # place X at position (x,y)
            symbol_terminal: None | QGraphicsTextItem
            if index in goals:
                symbol_terminal = self.addText('G')
            else:
                symbol_terminal = self.addText('X')
            assert type(symbol_terminal) is QGraphicsTextItem
            self.symbols_terminal.append(symbol_terminal)
            self.symbols_terminal[-1].setPos(x, y)
        for index in startings:
            # calculate position
            y = (index // w) * self.scene_width / w
            x = (index % w) * self.scene_height / h
            # place S at position (x,y)
            symbol_starting: None | QGraphicsTextItem = self.addText('S')
            assert type(symbol_starting) is QGraphicsTextItem
            self.symbols_starting.append(symbol_starting)
            self.symbols_starting[-1].setPos(x, y)
        self.update()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent | None) -> None:
        """
        This function determines whether a state has been clicked and if so which.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent or None
            The mouse press event.
        """
        assert event is not None
        # Gets mouse position within scene
        posX, posY = event.scenePos().x(), event.scenePos().y()
        # check if mouse within scene at all
        if (
            posX >= 0
            and posY >= 0
            and (posX <= self.scene_width)
            and posY <= self.scene_height
        ):
            # check if mouse position is on border (rel = relative pos in state)
            relX = (posX + self.main_window.border / 2) % self.main_window.width()
            relY = (posY + self.main_window.border / 2) % self.main_window.height()
            if relX > self.main_window.border + 0.5 and (
                relY > self.main_window.border + 0.5
            ):
                # translate to coordinate
                coordX = int(posY // self.main_window.height())
                coordY = int(posX // self.main_window.width())
                coordX = int(
                    posY // (self.scene_width / self.main_window.world['width'])
                )
                coordY = int(
                    posX // (self.scene_height / self.main_window.world['height'])
                )
                # calc index
                index = coordX * self.main_window.world['width'] + coordY
                # Console output of state coordinates
                coordText = '(%d, %d)' % (coordX, coordY)
                # highlight current state
                self.highlight_state(coordX, coordY)
                # update state info
                parent = self.parent()
                assert type(parent) is CentralPanel
                parent.change_state_information(coordText, index)


class Line(QtWidgets.QGraphicsLineItem):
    """
    An extended version of the standard QGraphicsLineItem which toggles
    the line color between gray and red on double click.

    Parameters
    ----------
    x1 : int
        X-Coordinate of first point of line.
    y1 : int
        Y-Coordinate of first point of line.
    x2 : int
        X-Coordinate of second point of line.
    y2 : int
        Y-Coordinate of second point of line.
    pen : QPen
        The pen that draws the line.
    main_window : MainWindow
        The gridworld editor's main window.

    """

    def __init__(
        self, x1: int, y1: int, x2: int, y2: int, pen: QPen, main_window: MainWindow
    ) -> None:
        super().__init__()
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
        self.highlighted = False
        self.main_window = main_window
        # determine the transition associated with this line
        self.transition = (0, 0)
        # vertical line
        if x1 == x2:
            coordX = y1 // self.main_window.height()
            coordY_left = (x1 - self.main_window.border) // self.main_window.width()
            coordY_right = (x1 + self.main_window.border) // self.main_window.width()
            self.transition = (
                int((coordX * self.main_window.world['width']) + coordY_left),
                int((coordX * self.main_window.world['width']) + coordY_right),
            )
        # horizontal line
        else:
            coordX_up = (y1 - self.main_window.border) // self.main_window.height()
            coordX_down = (y1 + self.main_window.border) // self.main_window.height()
            coordY = x1 // self.main_window.width()
            self.transition = (
                int((coordX_up * self.main_window.world['width']) + coordY),
                int((coordX_down * self.main_window.world['width']) + coordY),
            )

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent | None) -> None:
        """
        This function checks if a border has been double clicked.
        Un/highlights the border and edits the WORLD dict accordingly.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent or None
            The mouse press event.
        """
        self.main_window.changed = True
        action_coding = {
            1: 0,
            self.main_window.world['width']: 1,
            -1: 2,
            -self.main_window.world['width']: 3,
        }
        actions = [
            action_coding[self.transition[0] - self.transition[1]],
            action_coding[self.transition[1] - self.transition[0]],
        ]
        if not self.highlighted:
            self.setPen(QPen(QColorConstants.Red, self.main_window.border))
            # add invalid transitions
            self.main_window.world['invalid_transitions'].append(self.transition)
            self.main_window.world['invalid_transitions'].append(self.transition[::-1])
            self.main_window.world['sas'][self.transition[0], actions[0]] = np.eye(
                self.main_window.world['states']
            )[self.transition[0]]
            self.main_window.world['sas'][self.transition[1], actions[1]] = np.eye(
                self.main_window.world['states']
            )[self.transition[1]]
        else:
            self.setPen(QPen(QColorConstants.Gray, self.main_window.border))
            # remove invalid transitions
            self.main_window.world['invalid_transitions'].pop(
                self.main_window.world['invalid_transitions'].index(self.transition)
            )
            self.main_window.world['invalid_transitions'].pop(
                self.main_window.world['invalid_transitions'].index(
                    self.transition[::-1]
                )
            )
            self.main_window.world['sas'][self.transition[0], actions[0]] = np.eye(
                self.main_window.world['states']
            )[self.transition[1]]
            self.main_window.world['sas'][self.transition[1], actions[1]] = np.eye(
                self.main_window.world['states']
            )[self.transition[0]]
        # update the gridworld dictionary
        self.highlighted = not self.highlighted


class GridViewer(QGraphicsView):
    """
    The gridworld editor's grid view class.

    Parameters
    ----------
    scene : Grid
        The grid scene.
    parent : CentralPanel
        The grid viewer's parent window.
    main_window : MainWindow
        The gridworld editor's main window.

    """

    def __init__(
        self, scene: Grid, parent: CentralPanel, main_window: MainWindow
    ) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.setScene(scene)
        self.setParent(parent)
        self.setSceneRect(self.sceneRect())
        # Zoom counter and maximum zoom count
        self.zoom = 0
        self.max_zoom = 0
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """
        This function scales the grid view when the mouse's wheel is scrolled.

        Parameters
        ----------
        event : QWheelEvent or None
            This wheel event.
        """
        scene = self.scene()
        assert type(scene) is Grid
        assert type(event) is QWheelEvent
        # defines zoom factor depending on scroll direction
        if event.angleDelta().y() > 0:
            factor = 5 / 4
            self.zoom += 1
        else:
            factor = 4 / 5
            self.zoom -= 1
        # clip zoom level within valid range
        self.zoom = np.clip(self.zoom, 0, self.max_zoom + 1)
        # change scale
        if self.zoom > 0 and self.zoom <= self.max_zoom:
            self.scale(factor, factor)
        elif self.zoom == 0:
            # fit scene in view on maximum zoom out
            self.fitInView(
                scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
            )

        scene.update()

    def showEvent(self, event: QShowEvent | None) -> None:
        """
        This function fits the scene in view as soon as the view is shown.

        Parameters
        ----------
        event : QShowEvent or None
            This show event.
        """
        super().showEvent(event)
        scene = self.scene()
        viewport = self.viewport()
        assert type(scene) is Grid
        assert type(viewport) is QWidget
        self.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Determines maximum zoom count so you cannot zoom further when one
        # state fits the view
        maxFactor = max(
            scene.itemsBoundingRect().width() / viewport.rect().width(),
            scene.itemsBoundingRect().height() / viewport.rect().height(),
        )
        maxFactor *= min(
            viewport.rect().width() / self.main_window.width(),
            viewport.rect().height() / self.main_window.height(),
        )
        if maxFactor > 1:
            self.max_zoom = math.floor(math.log(maxFactor, 5 / 4))


class StateInformation(QWidget):
    """
    The gridworld editor's state information class.

    Parameters
    ----------
    parent : CentralPanel
        The state information's parent window.
    main_window : MainWindow
        The gridworld editor's main window.

    """

    def __init__(self, parent: CentralPanel, main_window: MainWindow) -> None:
        super().__init__(parent=parent)
        self.setParent(parent)
        self.main_window = main_window
        # WIDGETS
        # State Information
        heading = QLabel()
        heading.setText('STATE INFORMATION')
        # Index
        self.label_index = QLabel()
        self.index = 0
        # Coordinates
        self.label_coordinates = QLabel()
        # Reward
        self.label_reward = QLabel()
        self.label_reward.setText('Reward:')
        self.field_reward = QLineEdit()
        self.field_reward.setText('0')
        self.line_reward = QHBoxLayout()
        self.line_reward.addWidget(self.label_reward)
        self.line_reward.addWidget(self.field_reward)
        # Radio Button, Starting, Terminal, Goal
        self.label_state_type = QLabel('Set state type:')
        self.radio_button_goal = QRadioButton('Goal')
        self.radio_button_goal.setToolTip('Goal states are also terminal states!')
        self.radio_button_terminal = QRadioButton('Terminal')
        self.radio_button_start = QRadioButton('Starting')
        self.radio_button_none = QRadioButton('None')
        # Apply Button
        self.button_apply = QPushButton()
        self.button_apply.setText('Apply Changes')
        self.button_apply.clicked.connect(self.update_state)
        # Advanced Settings
        self.button_advanced = QPushButton()
        self.button_advanced.setText('Advanced Settings')
        self.button_advanced.clicked.connect(self.open_advanced)
        # Layout
        self.setLayout(QVBoxLayout())
        layout = self.layout()
        assert type(layout) is QVBoxLayout
        layout.addWidget(heading)
        layout.addWidget(self.label_index)
        layout.addWidget(self.label_coordinates)
        layout.addLayout(self.line_reward)
        layout.addWidget(self.label_state_type)
        layout.addWidget(self.radio_button_goal)
        layout.addWidget(self.radio_button_terminal)
        layout.addWidget(self.radio_button_start)
        layout.addWidget(self.radio_button_none)
        layout.addWidget(self.button_advanced)
        layout.addWidget(self.button_apply)

    def change_state_information(self, text_coordinates: str, index: int) -> None:
        """
        This function updates the state information panel
        according to the selected state.

        Parameters
        ----------
        text_coordinates : str
            The state coordinates.
        index : int
            The state's index.
        """
        self.label_index.setText('Index: %d' % int(index))
        self.index = int(index)
        self.label_coordinates.setText('Coordinates: %s' % text_coordinates)
        self.field_reward.setText(
            str(float(self.main_window.world['rewards'][int(index)]))
        )
        if self.main_window.world['terminals'][int(index)]:
            self.radio_button_terminal.click()
            if int(index) in self.main_window.world['goals']:
                self.radio_button_goal.click()
        elif index in self.main_window.world['starting_states']:
            self.radio_button_start.click()
        else:
            self.radio_button_none.click()

    def update_state(self) -> None:
        """
        This function updates the selected state's properties in the WORLD dictionary.
        """
        self.main_window.changed = True
        # update terminal status
        self.main_window.world['terminals'][int(self.index)] = int(
            self.radio_button_terminal.isChecked()
        )
        if self.radio_button_goal.isChecked():
            self.main_window.world['terminals'][int(self.index)] = 1
            self.main_window.world['goals'].append(int(self.index))
        elif int(self.index) in self.main_window.world['goals']:
            self.main_window.world['goals'].remove(int(self.index))
        # removes index from the starting state list
        if not self.radio_button_start.isChecked():
            self.main_window.world['starting_states'] = self.main_window.world[
                'starting_states'
            ][self.main_window.world['starting_states'] != int(self.index)]
        # add index as new starting state if it does not already exist
        elif self.radio_button_start.isChecked() and (
            int(self.index) not in self.main_window.world['starting_states']
        ):
            self.main_window.world['starting_states'] = np.append(
                self.main_window.world['starting_states'], int(self.index)
            )
        # update reward value
        self.main_window.world['rewards'][int(self.index)] = float(
            self.field_reward.text()
        )
        # mark state
        self.parent().parent().highlight_terminal_starting(  # type: ignore
            self.main_window.world['terminals'],
            self.main_window.world['starting_states'],
            self.main_window.world['goals'],
        )

    def open_advanced(self) -> None:
        """
        This funcion opens the advanced the setting menu.
        """
        self.main_window.index = int(self.index)
        self.advanced_settings = AdvancedSettingsWindow(self.main_window)
        self.advanced_settings.show()


class AdvancedSettingsWindow(QMainWindow):
    """
    The gridworld editor's advanced settings window class.

    Parameters
    ----------
    main_window : MainWindow
        The gridworld editor's main window.

    """

    def __init__(self, main_window: MainWindow, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.setWindowTitle('Advanced Settings')
        self.resize(500, 500)
        self.advanced_widget = AdvancedSettingsWidget(main_window)
        self.setCentralWidget(self.advanced_widget)


class AdvancedSettingsWidget(QWidget):
    """
    The widget that handles the table for the state transition probabilities.

    Parameters
    ----------
    main_window : MainWindow
        The gridworld editor's main window.

    """

    def __init__(self, main_window: MainWindow) -> None:
        super().__init__()
        self.main_window = main_window
        # The table with 4 columns and as many roows as there are states
        self.table_widget = QTableWidget(self.main_window.world['states'], 4, self)
        self.table_widget.setHorizontalHeaderLabels(
            [str(i) for i in range(self.table_widget.columnCount())]
        )
        self.table_widget.setVerticalHeaderLabels(
            [str(i) for i in range(self.table_widget.rowCount())]
        )
        # This fills the table with QLineEdits,
        # which are filled with the data from the ["sas"] array
        for column in range(self.table_widget.columnCount()):
            for row in range(self.table_widget.rowCount()):
                cellValue = QLineEdit()
                cellValue.setText(
                    str(
                        float(
                            self.main_window.world['sas'][int(self.main_window.index)][
                                column
                            ][row]
                        )
                    )
                )
                cellValue.setFrame(False)
                self.table_widget.setCellWidget(row, column, cellValue)
        # make save button
        button_apply = QPushButton()
        button_apply.setText('Apply Changes')
        button_apply.clicked.connect(self.apply_changes)
        # set layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.table_widget)
        layout.addWidget(button_apply)

    def apply_changes(self) -> None:
        """
        This function closes the advanced settings window and applies the change made
        """
        # Array to save the changed transition probabilities and
        # check if the entered data is valid - smaller than 1
        transition_probabilities = np.zeros((4, self.main_window.world['states']))
        for column in range(self.table_widget.columnCount()):
            for row in range(self.table_widget.rowCount()):
                cell_widget = self.table_widget.cellWidget(row, column)
                assert type(cell_widget) is QLineEdit
                transition_probabilities[column][row] = float(cell_widget.text())
        # ensure that probabilities sum to one
        transition_probabilities /= np.sum(transition_probabilities, axis=1).reshape(
            (4, 1)
        )
        for state in range(self.main_window.world['states']):
            for action in range(4):
                self.main_window.world['sas'][self.main_window.index][action][state] = (
                    transition_probabilities[action][state]
                )
        self.parent().close()  # type: ignore


class NewDialog(QDialog):
    """
    This class implements the Gridworld editor's new gridworld dialog.

    Parameters
    ----------
    parent : MainWindow
        The new gridworld dialog's parent window.

    """

    def __init__(self, parent: MainWindow) -> None:
        super().__init__(parent)
        self.setParent(parent)
        # prepare dialog
        self.setWindowTitle('New Gridworld')
        layout = QVBoxLayout()
        self.QBtn = (
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box = QDialogButtonBox(self.QBtn)
        self.button_box.accepted.connect(self.apply)
        self.button_box.rejected.connect(self.reject)
        # parameter input
        self.validator = QIntValidator(1, 100)
        self.label_width = QLabel('Set gridworld width:')
        self.input_width = QLineEdit()
        self.input_width.setPlaceholderText('Width/Columns(y)')
        self.input_width.setValidator(self.validator)
        self.label_height = QLabel('Set gridworld height:')
        self.input_height = QLineEdit()
        self.input_height.setPlaceholderText('Height/Rows(x)')
        self.input_height.setValidator(self.validator)
        self.label_state_type = QLabel('Set default state type:')
        self.radio_button_start = QRadioButton('Starting')
        self.radio_button_none = QRadioButton('None')
        self.radio_button_none.click()
        # add elements to panel
        layout.addWidget(self.label_width)
        layout.addWidget(self.input_width)
        layout.addWidget(self.label_height)
        layout.addWidget(self.input_height)
        layout.addWidget(self.label_state_type)
        layout.addWidget(self.radio_button_start)
        layout.addWidget(self.radio_button_none)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def apply(self) -> None:
        """
        This functions generates a new gridworld with the
        entered parameters and updates the GUI.
        """
        # make gridworld
        width, height = int(self.input_width.text()), int(self.input_height.text())
        world = gwt.make_gridworld(height, width, invalid_transitions=[])
        if not self.radio_button_start.isChecked():
            world['starting_states'] = np.array([], dtype=int)
        # update GUI
        parent = self.parent()
        assert type(parent) is MainWindow

        # breaks editor; fix later
        # parent.world = world
        # parent.changed = False
        # parent.file = None
        # parent.resize(50, 50)
        # parent.border = 5
        # parent.index = 0
        # parent.setCentralWidget(None)
        # parent.central_panel = CentralPanel(parent)
        # parent.setCentralWidget(parent.central_panel)
        # parent.resize(1000, 563)
        # self.accept()

        # dirty solution as the previous currently breaks the editor
        main = MainWindow(world)
        main.show()
        self.setParent(main)
        parent.close()
        self.accept()


class TemplateDialog(QDialog):
    """
    This class implements the Gridworld editor's template dialog.
    According to the choice of template the dialog is generated dynamically.

    Parameters
    ----------
    parent : MainWindow
        The template dialog's parent window.
    maze : str
        The template type (i.e., 't_maze', 'double_t_maze',
        'two_sided_t_maze', '8_maze', 'two_choice_t_maze', 'detour_maze').

    """

    def __init__(self, parent: MainWindow, maze: str) -> None:
        super().__init__(parent)
        self.setParent(parent)
        self.maze = maze
        # define template parameters
        self.templates = {
            't_maze': {
                'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                'stem_length': {
                    'type': int,
                    'range': [1, None],
                    'label': 'Stem Length',
                },
                'goal_arm': {
                    'type': list,
                    'range': ['left', 'right'],
                    'label': 'Goal Arm',
                },
                'reward': {'type': float, 'range': [None, None], 'label': 'Reward'},
            },
            'double_t_maze': {
                'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                'stem_length': {
                    'type': int,
                    'range': [1, None],
                    'label': 'Stem Length',
                },
                'goal_arm': {
                    'type': list,
                    'range': ['left-left', 'left-right', 'right-left', 'right-right'],
                    'label': 'Goal Arm',
                },
                'reward': {'type': float, 'range': [None, None], 'label': 'Reward'},
            },
            'two_sided_t_maze': {
                'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                'stem_length': {
                    'type': int,
                    'range': [1, None],
                    'label': 'Stem Length',
                },
                'goal_arm': {
                    'type': list,
                    'range': ['left-left', 'left-right', 'right-left', 'right-right'],
                    'label': 'Goal Arm',
                },
                'reward': {'type': float, 'range': [None, None], 'label': 'Reward'},
            },
            '8_maze': {
                'center_height': {'type': int, 'range': [1, None], 'label': 'Height'},
                'lap_width': {'type': int, 'range': [1, None], 'label': 'Lap Width'},
                'goal_location': {
                    'type': list,
                    'range': ['left', 'right'],
                    'label': 'Goal Location',
                },
                'reward': {'type': float, 'range': [None, None], 'label': 'Reward'},
            },
            'two_choice_t_maze': {
                'center_height': {'type': int, 'range': [1, None], 'label': 'Height'},
                'lap_width': {'type': int, 'range': [1, None], 'label': 'Lap Width'},
                'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                'chirality': {
                    'type': list,
                    'range': ['left', 'right'],
                    'label': 'Chirality',
                },
                'goal_location': {
                    'type': list,
                    'range': ['left', 'right'],
                    'label': 'Goal Location',
                },
                'reward': {'type': float, 'range': [None, None], 'label': 'Reward'},
            },
            'detour_maze': {
                'width_small': {
                    'type': int,
                    'range': [1, None],
                    'label': 'Width Small',
                },
                'height_small': {
                    'type': int,
                    'range': [1, None],
                    'label': 'Height Small',
                },
                'width_large': {
                    'type': int,
                    'range': [1, None],
                    'label': 'Width Large',
                },
                'height_large': {
                    'type': int,
                    'range': [1, None],
                    'label': 'Height Large',
                },
                'reward': {'type': float, 'range': [None, None], 'label': 'Reward'},
            },
        }
        # define template labels
        self.template_labels = {
            't_maze': 'T-maze',
            'double_t_maze': 'Double T-maze',
            'two_sided_t_maze': 'Two-sided T-maze',
            '8_maze': '8-maze',
            'two_choice_t_maze': 'Two-choice T-maze',
            'detour_maze': 'Detour Maze',
        }
        # register functions
        self.template_functions = {
            't_maze': gwt.make_t_maze,
            'double_t_maze': gwt.make_double_t_maze,
            'two_sided_t_maze': gwt.make_two_sided_t_maze,
            '8_maze': gwt.make_8_maze,
            'two_choice_t_maze': gwt.make_two_choice_t_maze,
            'detour_maze': gwt.make_detour_maze,
        }
        # prepare dialog
        self.setWindowTitle('Configure %s template!' % self.template_labels[self.maze])
        layout = QVBoxLayout()
        self.param_layout = QVBoxLayout()
        self.QBtn = (
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box = QDialogButtonBox(self.QBtn)
        self.button_box.accepted.connect(self.apply)
        self.button_box.rejected.connect(self.reject)
        # generate template specific layout
        self.labels: dict[str, QLabel] = {}
        self.params: dict[str, QLineEdit | QComboBox] = {}
        for param in self.templates[self.maze]:
            # param label
            label = str(self.templates[self.maze][param]['label'])
            self.labels[param] = QLabel(label)
            # param input
            if self.templates[self.maze][param]['type'] in [int, float]:
                self.params[param] = QLineEdit()
            elif self.templates[self.maze][param]['type'] is list:
                param_range = self.templates[self.maze][param]['range']
                assert type(param_range) is list
                box = QComboBox()
                box.addItems(param_range)
                self.params[param] = box
            # add to layout
            self.param_layout.addWidget(self.labels[param])
            self.param_layout.addWidget(self.params[param])
        layout.addLayout(self.param_layout)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def apply(self) -> None:
        """
        This functions generates a new gridworld from a
        selected template and updates the GUI.
        """
        # retrieve parameters
        params = {}
        for param in self.params:
            if self.templates[self.maze][param]['type'] in [int, float]:
                line = self.params[param]
                assert type(line) is QLineEdit
                params[param] = self.templates[self.maze][param]['type'](line.text())  # type: ignore
            else:
                box = self.params[param]
                assert type(box) is QComboBox
                params[param] = box.currentText()
        # generate gridworld from template
        world = self.template_functions[self.maze](**params)  # type: ignore
        # invalid parameters yield an empty dictionary
        if len(world) == 0:
            QMessageBox.critical(
                self, 'Invalid Parameters!', 'Please adjust template parameters.'
            )
            return
        # world['starting_states'] = np.array([]).astype(int)
        # update GUI
        parent = self.parent()
        assert type(parent) is MainWindow
        # parent.world = world
        # parent.changed = False
        # parent.file = None
        # parent.index = 0
        # parent.central_panel = CentralPanel(parent)
        # parent.setCentralWidget(parent.central_panel)
        # self.accept()

        # dirty solution as the previous currently breaks the editor
        main = MainWindow(world)
        main.show()
        self.setParent(main)
        parent.close()
        self.accept()


class UnsavedChangesDialog(QDialog):
    """
    The gridworld editor's unsaved changes dialog class.

    Parameters
    ----------
    parent : MainWindow
        The unsaved changes dialog's parent window.

    """

    def __init__(self, parent: MainWindow) -> None:
        super().__init__(parent)
        self.setParent(parent)
        layout = QGridLayout()
        self.setWindowTitle('Unsaved Changes')
        # prepare elements
        self.label_dialog = QLabel()
        self.label_dialog.setText(
            'The file has been modified. Do you want to save the changes?'
        )
        self.button_yes = QPushButton('Yes')
        self.button_no = QPushButton('No')
        self.button_cancel = QPushButton('Cancel')
        # make layout
        layout.addWidget(self.label_dialog, 0, 0, 1, 3)
        layout.addWidget(self.button_yes, 1, 0, 1, 1)
        layout.addWidget(self.button_no, 1, 1, 1, 1)
        layout.addWidget(self.button_cancel, 1, 2, 1, 1)
        self.button_yes.clicked.connect(self.save_current_file)
        self.button_no.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)
        self.setLayout(layout)
        self.changed = False

    def save_current_file(self) -> None:
        """
        This function saves the gridworld in its currently selected file.
        If none is currently selected the user will be asked to choose one.
        """
        parent = self.parent()
        assert type(parent) is MainWindow
        self.hide()
        parent.save()
        if not self.changed:
            self.accept()
        else:
            self.reject()


class InfoDialog(QDialog):
    """
    The gridworld editor's info dialog class.

    Parameters
    ----------
    parent : MainWindow
        The info dialog's parent window.

    """

    def __init__(self, parent: MainWindow) -> None:
        super().__init__(parent)
        self.setParent(parent)
        layout = QGridLayout()
        self.setWindowTitle('Info')
        self.label_dev = QLabel(
            'Developed by : William Forchap, Kilian Kandt, Marius Tenhumberg,'
            ' Chuan Jin, Umut Yilmaz, Nicolas Diekmann'
        )
        self.label_sup = QLabel('Supervised by : Nicolas Diekmann')
        self.button_ok = QPushButton('Ok')
        layout.addWidget(self.label_dev, 0, 0, 1, 1)
        layout.addWidget(self.label_sup, 1, 0, 1, 1)
        layout.addWidget(self.button_ok, 2, 0, 1, 1)
        self.button_ok.clicked.connect(self.accept)
        self.setLayout(layout)


def main() -> None:
    app = QApplication(sys.argv)
    a = MainWindow()
    a.show()
    app.exec()


if __name__ == '__main__':
    main()
