# basic imports
import sys
import math
import pickle
import numpy as np
# Qt imports
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPen, QBrush, QColor, QIntValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QAction, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, \
    QLabel, QRadioButton, QTableWidget, QGraphicsScene, QLineEdit, QGraphicsView, QSplitter, QPushButton, QFileDialog, \
    QDialogButtonBox, QMessageBox, QComboBox
from PyQt5 import QtWidgets
# framework imports
import cobel.misc.gridworld_tools as gwt
import cobel.misc.gridworld_export as gwe


class MainWindow(QMainWindow):

    def __init__(self, world: None | dict = None, *args, **kwargs):
        '''
        The gridworld editor's main window class.

        Parameters
        ----------
        world :                             The gridworld dictionary (5 x 5 open field by default).

        Returns
        ----------
        None
        '''
        super().__init__(*args, **kwargs)
        # gridworld
        self.world = gwt.make_open_field(5, 5) if world is None else world
        self.changed = False
        self.file = None
        self.height, self.width, self.border = 50, 50, 5
        self.index = 0
        # init GUI
        self.setWindowTitle('Gridworld Editor')
        self.create_MenuBar()
        self.central_panel = CentralPanel(self)
        self.setCentralWidget(self.central_panel)
        # window size
        self.resize(1000, 563)

    def create_MenuBar(self):
        '''
        This function creates the menu bar.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # Define actions
        # new
        self.action_new = QAction('&New', self)
        self.action_new.setShortcut("Ctrl+N")
        self.action_new.triggered.connect(self.new)
        # open
        self.action_open = QAction('&Open', self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.triggered.connect(self.load)
        # save
        self.action_save = QAction('&Save', self)
        self.action_save.setShortcut("Ctrl+S")
        self.action_save.triggered.connect(self.save)
        # save as
        self.action_save_as = QAction('&Save as', self)
        self.action_save_as.setShortcut("Ctrl+Shift+S")
        self.action_save_as.triggered.connect(self.save_as)
        # quit
        self.action_quit = QAction('&Quit', self)
        self.action_quit.setShortcut("Ctrl+Q")
        self.action_quit.triggered.connect(self.close)
        # export
        self.action_export = QAction('&Export')
        self.action_export.setShortcut("Ctrl+E")
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
        self.action_info.setShortcut("Ctrl+I")
        self.action_info.triggered.connect(self.info)
        # create menu nar
        self.menu_bar = self.menuBar()
        # add menus to menu bar
        self.file_menu = self.menu_bar.addMenu('&File')
        self.info_menu = self.menu_bar.addMenu('&Info')
        self.tools_menu = self.menu_bar.addMenu('&Templates')
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

    def open_tmaze_dialog(self):
        '''
        This function opens the T-maze template dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.temps_open = TemplateDialog(self, 't_maze')
        self.temps_open.show()

    def open_eightmaze_dialog(self):
        '''
        This function opens the 8-maze template dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.temps_open = TemplateDialog(self, '8_maze')
        self.temps_open.show()

    def open_twochoice_dialog(self):
        '''
        This function opens the Two-choice T-maze template dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.temps_open = TemplateDialog(self, 'two_choice_t_maze')
        self.temps_open.show()

    def open_detour_dialog(self):
        '''
        This function opens the detour maze template dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.temps_open = TemplateDialog(self, 'detour_maze')
        self.temps_open.show()

    def open_twosided_tmaze_dialog(self):
        '''
        This function opens the Two-sided T-maze template dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.temps_open = TemplateDialog(self, 'two_sided_t_maze')
        self.temps_open.show()

    def open_double_tmaze_dialog(self):
        '''
        This function opens the Double T-maze template dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.temps_open = TemplateDialog(self, 'double_t_maze')
        self.temps_open.show()

    def new(self):
        '''
        This function opens the new gridworld dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        if not self.unsaved_changes():
            self.new_open = NewDialog(self)
            self.new_open.show()

    def load(self):
        '''
        This function asks the user to select a file and opens it.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        if not self.unsaved_changes():
            file_name = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Pickle files (*.pkl)")[0]
            print('Try loading file: ', file_name)
            try:
                self.world = pickle.load(open(file_name, 'rb'))
                self.changed = False
                self.file = None
                self.index = 0
                self.central_panel = CentralPanel(self)
                self.setCentralWidget(self.central_panel)
                print('Successfully loaded file. ')
            except:
                print('Couldn\'t load file.')

    def save_as(self):
        '''
        This function saves the current gridworld in a file.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        file_name = QFileDialog.getSaveFileName(self, 'Open file', 'c:\\', "Pickle files (*.pkl)")[0]
        print('Try saving file: ', file_name)
        try:
            pickle.dump(self.world, open(file_name, 'wb'))
            self.changed = False
            self.file = file_name
            print('Successfully saved file. ')
        except:
            print('Couldn\'t save file.')

    def save(self):
        '''
        This function saves the current gridworld.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        if self.file is None:
            self.save_as()
        else:
            print('Try saving file: ', self.file)
            try:
                pickle.dump(self.world, open(self.file, 'wb'))
                self.changed = False
                print('Successfully saved file. ')
            except:
                print('Couldn\'t save file.')

    def export(self):
        '''
        This function exports the current gridworld into a Wavefront obj file.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        file_name = QFileDialog.getSaveFileName(self, 'export as 3D-Model', '', "Obj files (*.obj)")[0]
        if file_name:
            invalid_transitions, width, height = self.world['invalid_transitions'], self.world['width'], self.world['height']
            wall_info, pillars = gwe.retrieve_wall_info(invalid_transitions, width, height, state_size=1)
            walls = gwe.generate_walls(wall_info, pillars, width, height, state_size=1, wall_height=1, wall_depth=0.2)
            gwe.export_as_obj(walls, width, height, state_size=1, file_name=file_name)
        else:
            print("filename not found")

    def unsaved_changes(self):
        '''
        This function checks for unsaved changes and informs the user accordingly.

        Parameters
        ----------
        None

        Returns
        ----------
        flag :                              True, if unsaved changes were found.
        '''
        # Returns False if there are no unsaved changes/if these should be ignored
        if self.changed:
            print('Unsaved changes.')
            self.unsaved_changes_dialog = UnsavedChangesDialog(self)
            if self.unsaved_changes_dialog.exec():
                return False
            else:
                return True

        return False

    def info(self):
        '''
        This functions opens the info dialog.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.info_dialog = InfoDialog(self)
        self.info_dialog.exec()

    def closeEvent(self, event: QEvent):
        '''
        This function closes the program.

        Parameters
        ----------
        event :                             The close event.

        Returns
        ----------
        None
        '''
        if not self.unsaved_changes():
            super().closeEvent(event)
        else:
            event.ignore()


class CentralPanel(QWidget):

    def __init__(self, parent: MainWindow):
        '''
        The widget that handles the two main widgets (grid and state information).

        Parameters
        ----------
        parent :                            The central panel's parent window (i.e., the main window).

        Returns
        ----------
        None
        '''
        super().__init__()
        self.parent = parent
        # Grid and View
        self.scene = Grid(self, parent)
        self.view = GridViewer(self.scene, self, parent)
        self.scene.highlight_terminal_starting(parent.world['terminals'], parent.world['starting_states'])
        # Sidebar
        self.state_menu = StateInformation(self, parent)
        # splitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.state_menu)  # sidebar here
        self.splitter.setStretchFactor(0, 10)
        # add splitter(Grid & InfoBar) to layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)

    def change_state_information(self, textCoord: str, index: int):
        '''
        This function updates the state information panel.

        Parameters
        ----------
        textCoord :                         The state's coordinates.
        index :                             The state's index.

        Returns
        ----------
        None
        '''
        self.state_menu.change_state_information(textCoord, index)

    def highlight_terminal_starting(self, terminals: np.ndarray, startings: list):
        '''
        This function marks states according to whether they are terminal or starting states.

        Parameters
        ----------
        terminal :                          A numpy array containg the gridworld's terminal state's indeces.
        startings :                         A list containing the gridworld's starting state's indeces.

        Returns
        ----------
        None
        '''
        self.scene.highlight_terminal_starting(terminals, startings)


class Grid(QGraphicsScene):

    def __init__(self, parent: CentralPanel, main_window: MainWindow):
        '''
        This class presents the grid of the gridworld.

        Parameters
        ----------
        parent :                            The grid's parent window.
        main_window :                       The gridworld editor's main window.

        Returns
        ----------
        None
        '''
        super().__init__(parent)
        self.parent = parent
        self.main_window = main_window
        self.lines = []
        self.highlight = self.addRect(self.main_window.border / 2, self.main_window.border / 2,
                                      self.main_window.width - self.main_window.border, self.main_window.height - self.main_window.border,
                                      QPen(QColor(125, 175, 240, 0), 0), QBrush(QColor(125, 175, 240, 0)))
        self.height = self.main_window.world['height'] * self.main_window.height
        self.width = self.main_window.world['width'] * self.main_window.width
        self.pen = QPen(QColor(125, 175, 240, 125), 0)
        self.brush = QBrush(QColor(125, 175, 240, 125))
        self.draw_grid()
        self.symbols_terminal, self.symbols_starting = [], []

    def draw_grid(self):
        '''
        This function draws the grid.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        pen_lines = QPen(Qt.gray, self.main_window.border)
        pen_border = QPen(Qt.black, self.main_window.border)
        # lines parallel to GUI x-axis (vertical)
        for column in range(1, self.main_window.world['width']):
            x = column * self.main_window.width
            for row in range(0, self.main_window.world['height']):
                y = row * self.main_window.height
                line = Line(x, y, x, y + self.main_window.height, pen_lines, self.main_window)
                if line.transition in self.main_window.world['invalid_transitions']:
                    line.highlighted = True
                    line.setPen(QPen(Qt.red, self.main_window.border))
                self.lines.append(self.addItem(line))
        # lines parallel to GUI y-axis (horizontal)
        for row in range(1, self.main_window.world['height']):
            y = row * self.main_window.height
            for column in range(0, self.main_window.world['width']):
                x = column * self.main_window.width
                line = Line(x, y, x + self.main_window.width, y, pen_lines, self.main_window)
                if line.transition in self.main_window.world['invalid_transitions']:
                    line.highlighted = True
                    line.setPen(QPen(Qt.red, self.main_window.border))
                self.lines.append(self.addItem(line))
        # Outer border
        self.lines.append(self.addLine(0, 0, self.width, 0, pen_border))
        self.lines.append(self.addLine(0, 0, 0, self.height, pen_border))
        self.lines.append(self.addLine(0, self.height, self.width, self.height, pen_border))
        self.lines.append(self.addLine(self.width, 0, self.width, self.height, pen_border))

    def set_visible(self, visible: bool = True):
        '''
        This function sets the visibility of the grid.

        Parameters
        ----------
        visible :                           Flag determining the visibility of the grid.

        Returns
        ----------
        None
        '''
        for line in self.lines:
            line.setVisible(visible)

    def delete_grid(self):
        '''
        This function deletes the grid.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        for line in self.lines:
            self.removeItem(line)
        del self.lines[:]

    def set_opacity(self, opacity: float):
        '''
        This function sets the opacity of the grid lines.

        Parameters
        ----------
        opacity :                           The grid line opacity.

        Returns
        ----------
        None
        '''
        for line in self.lines:
            line.setOpacity(opacity)

    def highlight_state(self, x: int, y: int):
        '''
        This function highlights a state.

        Parameters
        ----------
        x :                                 The state's x coordinate.
        y :                                 The state's y coordinate.

        Returns
        ----------
        None
        '''
        # removes pieces old highlight
        self.removeItem(self.highlight)
        self.highlight = self.addRect(y * self.main_window.width + self.main_window.border / 2, x * self.main_window.height + self.main_window.border / 2,
                                      self.main_window.width - self.main_window.border, self.main_window.height - self.main_window.border,
                                      self.pen, self.brush)
        # removes artefacts of old highlights
        self.update()

    def highlight_terminal_starting(self, terminals: np.ndarray, startings: list):
        '''
        This function marks states according to whether they are terminal or starting states.

        Parameters
        ----------
        terminals :                         A numpy array containing the terminal states' indeces.
        startings :                         A list containing the starting states' indeces.

        Returns
        ----------
        None
        '''
        # delete all symbols created before and then empty the array
        for index in range(len(self.symbols_terminal)):
            self.removeItem(self.symbols_terminal[-1])
            self.symbols_terminal.pop(-1)
        for index in range(len(self.symbols_starting)):
            self.removeItem(self.symbols_starting[-1])
            self.symbols_starting.pop(-1)
        # create symbol at position x,y for all terminals and startings
        for index in np.arange(terminals.shape[0])[terminals == 1]:
            # calculate position
            y = (index // self.main_window.world['width']) * self.main_window.width
            x = (index % self.main_window.world['width']) * self.main_window.height
            # place X at position (x,y)
            self.symbols_terminal.append(self.addText('X'))
            self.symbols_terminal[-1].setPos(x, y)
        for index in startings:
            # calculate position
            y = (index // self.main_window.world['width']) * self.main_window.width
            x = (index % self.main_window.world['width']) * self.main_window.height
            # place S at position (x,y)
            self.symbols_starting.append(self.addText('S'))
            self.symbols_starting[-1].setPos(x, y)
        self.update()

    def mousePressEvent(self, event: QEvent):
        '''
        This function determines whether a state has been clicked and if so which.

        Parameters
        ----------
        event :                             The mouse press event.

        Returns
        ----------
        None
        '''
        # Gets mouse position within scene
        posX, posY = event.scenePos().x(), event.scenePos().y()
        # check if mouse within scene at all
        if posX >= 0 and posY >= 0 and posX <= self.width and posY <= self.height:
            # check if mouse position is on border (rel = relative pos in state)
            relX = (posX + self.main_window.border / 2) % self.main_window.width
            relY = (posY + self.main_window.border / 2) % self.main_window.height
            if relX > self.main_window.border + 0.5 and relY > self.main_window.border + 0.5:
                # translate to coordinate
                coordX, coordY = int(posY // self.main_window.height), int(posX // self.main_window.width)
                # calc index
                index = coordX * self.main_window.world['width'] + coordY
                # Console output of state coordinates
                coordText = '(%d, %d)' % (coordX, coordY)
                # highlight current state
                self.highlight_state(coordX, coordY)
                # update state info
                self.parent.change_state_information(coordText, index)


class Line(QtWidgets.QGraphicsLineItem):

    def __init__(self, x1: int, y1: int, x2: int, y2: int, pen: QPen, main_window: MainWindow):
        '''
        An extended version of the standard QGraphicsLineItem which toggles the line color between gray and red on double click.

        Parameters
        ----------
        x1 :                                X-Coordinate of first point of line.
        y1 :                                Y-Coordinate of first point of line.
        x2 :                                X-Coordinate of last point of line.
        y2 :                                Y-Coordinate of last point of line.
        pen :                               The pen that draws the line.
        main_window :                       The gridworld editor's main window.

        Returns
        ----------
        None
        '''
        super().__init__()
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
        self.highlighted = False
        self.main_window = main_window
        # determine the transition associated with this line
        self.transition = (0, 0)
        # vertical line
        if x1 == x2:
            coordX = y1 // self.main_window.height
            coordY_left = (x1 - self.main_window.border) // self.main_window.width
            coordY_right = (x1 + self.main_window.border) // self.main_window.width
            self.transition = (int((coordX * self.main_window.world['width']) + coordY_left), int((coordX * self.main_window.world['width']) + coordY_right))
        # horizontal line
        else:
            coordX_up = (y1 - self.main_window.border) // self.main_window.height
            coordX_down = (y1 + self.main_window.border) // self.main_window.height
            coordY = x1 // self.main_window.width
            self.transition = (int((coordX_up * self.main_window.world['width']) + coordY), int((coordX_down * self.main_window.world['width']) + coordY))

    def mousePressEvent(self, event: QEvent):
        '''
        This function checks if a border has been double clicked. Un/highlights the border and edits the WORLD dict accordingly.

        Parameters
        ----------
        event :                             The mouse press event.

        Returns
        ----------
        None
        '''
        self.main_window.changed = True
        action_coding = {1: 0, self.main_window.world['width']: 1, -1: 2, -self.main_window.world['width']: 3}
        actions = [action_coding[self.transition[0] - self.transition[1]], action_coding[self.transition[1] - self.transition[0]]]
        if not self.highlighted:
            self.setPen(QPen(Qt.red, self.main_window.border))
            # add invalid transitions
            self.main_window.world['invalid_transitions'].append(self.transition)
            self.main_window.world['invalid_transitions'].append(self.transition[::-1])
            self.main_window.world['sas'][self.transition[0], actions[0]] = np.eye(self.main_window.world['states'])[self.transition[0]]
            self.main_window.world['sas'][self.transition[1], actions[1]] = np.eye(self.main_window.world['states'])[self.transition[1]]
        else:
            self.setPen(QPen(Qt.gray, self.main_window.border))
            # remove invalid transitions
            self.main_window.world['invalid_transitions'].pop(self.main_window.world['invalid_transitions'].index(self.transition))
            self.main_window.world['invalid_transitions'].pop(self.main_window.world['invalid_transitions'].index(self.transition[::-1]))
            self.main_window.world['sas'][self.transition[0], actions[0]] = np.eye(self.main_window.world['states'])[self.transition[1]]
            self.main_window.world['sas'][self.transition[1], actions[1]] = np.eye(self.main_window.world['states'])[self.transition[0]]
        # update the gridworld dictionary
        self.highlighted = not self.highlighted


class GridViewer(QGraphicsView):

    def __init__(self, scene: Grid, parent: CentralPanel, main_window: MainWindow):
        '''
        The gridworld editor's grid view class.

        Parameters
        ----------
        scene :                             The grid scene.
        parent :                            The grid viewer's parent window.
        main_window :                       The gridworld editor's main window.

        Returns
        ----------
        None
        '''
        super().__init__(parent)
        self.main_window = main_window
        self.scene = scene
        self.setScene(self.scene)
        self.parent = parent
        self.setSceneRect(self.sceneRect())
        # Zoom counter and maximum zoom count
        self.zoom = 0
        self.max_zoom = 0
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def wheelEvent(self, event: QEvent):
        '''
        This function scales the grid view when the mouse's wheel is scrolled.

        Parameters
        ----------
        event :                             This wheel event.

        Returns
        ----------
        None
        '''
        # defines zoom factor depending on scroll direction
        if event.angleDelta().y() > 0:
            factor = 5/4
            self.zoom += 1
        else:
            factor = 4/5
            self.zoom -= 1
        # clip zoom level within valid range
        self.zoom = np.clip(self.zoom, 0, self.max_zoom + 1)
        # change scale
        if self.zoom > 0 and self.zoom <= self.max_zoom:
            self.scale(factor, factor)
        elif self.zoom == 0:
            # fit scene in view on maximum zoom out
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        self.scene.update()

    def showEvent(self, event):
        '''
        This function fits the scene in view as soon as the view is shown.

        Parameters
        ----------
        event :                             This show event.

        Returns
        ----------
        None
        '''
        super().showEvent(event)
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        # Determines maximum zoom count so you cannot zoom further when one
        # state fits the view
        maxFactor = max(self.scene.itemsBoundingRect().width()/self.viewport().rect().width(),
                        self.scene.itemsBoundingRect().height()/self.viewport().rect().height())
        maxFactor *= min(self.viewport().rect().width()/self.main_window.width,
                         self.viewport().rect().height()/self.main_window.height)
        if maxFactor > 1:
            self.max_zoom = math.floor(math.log(maxFactor, 5/4))


class StateInformation(QWidget):

    def __init__(self, parent: CentralPanel, main_window: MainWindow):
        '''
        The gridworld editor's state information class.

        Parameters
        ----------
        parent :                            The state information's parent window.
        main_window :                       The gridworld editor's main window.

        Returns
        ----------
        None
        '''
        super().__init__(parent=parent)
        self.parent = parent
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
        self.layout().addWidget(heading)
        self.layout().addWidget(self.label_index)
        self.layout().addWidget(self.label_coordinates)
        self.layout().addLayout(self.line_reward)
        self.layout().addWidget(self.label_state_type)
        self.layout().addWidget(self.radio_button_terminal)
        self.layout().addWidget(self.radio_button_start)
        self.layout().addWidget(self.radio_button_none)
        self.layout().addWidget((self.button_advanced))
        self.layout().addWidget(self.button_apply)

    def change_state_information(self, text_coordinates: str, index: int):
        '''
        This function updates the state information panel according to the selected state.

        Parameters
        ----------
        text_coordinates :                  The state coordinates.
        index :                             The state's index.

        Returns
        ----------
        None
        '''
        self.label_index.setText('Index: %d' % int(index))
        self.index = int(index)
        self.label_coordinates.setText('Coordinates: %s' % text_coordinates)
        self.field_reward.setText(str(float(self.main_window.world['rewards'][int(index)])))
        if self.main_window.world['terminals'][int(index)]:
            self.radio_button_terminal.click()
        elif index in self.main_window.world['starting_states']:
            self.radio_button_start.click()
        else:
            self.radio_button_none.click()

    def update_state(self):
        '''
        This function updates the selected state's properties in the WORLD dictionary.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.main_window.changed = True
        # update terminal status
        self.main_window.world['terminals'][int(self.index)] = int(self.radio_button_terminal.isChecked())
        # removes index from the starting state list
        if not self.radio_button_start.isChecked():
            self.main_window.world['starting_states'] = self.main_window.world['starting_states'][self.main_window.world['starting_states'] != int(self.index)]
        # add index as new starting state if it does not already exist
        elif self.radio_button_start.isChecked() and not (int(self.index) in self.main_window.world['starting_states']):
            self.main_window.world['starting_states'] = np.append(self.main_window.world['starting_states'], int(self.index))
        # update reward value
        self.main_window.world['rewards'][int(self.index)] = float(self.field_reward.text())
        # mark state
        self.parent.highlight_terminal_starting(self.main_window.world['terminals'], self.main_window.world['starting_states'])

    def open_advanced(self):
        '''
        This funcion opens the advanced the setting menu.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        '''
        This function opens the advanced settings menu.
        '''
        self.main_window.index = int(self.index)
        self.advanced_settings = AdvancedSettingsWindow(self.main_window)
        self.advanced_settings.show()


class AdvancedSettingsWindow(QMainWindow):

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        '''
        The gridworld editor's advanced settings window class.

        Parameters
        ----------
        main_window :                       The gridworld editor's main window.

        Returns
        ----------
        None
        '''
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.setWindowTitle('Advanced Settings')
        self.resize(500, 500)
        self.advanced_widget = AdvancedSettingsWidget(main_window)
        self.setCentralWidget(self.advanced_widget)


class AdvancedSettingsWidget(QWidget):

    def __init__(self, main_window: MainWindow):
        '''
        The widget that handles the table for the state transition probabilities.

        Parameters
        ----------
        main_window :                       The gridworld editor's main window.

        Returns
        ----------
        None
        '''
        super().__init__()
        self.main_window = main_window
        # The table with 4 columns and as many roows as there are states
        self.table_widget = QTableWidget(self.main_window.world['states'], 4, self)
        self.table_widget.setHorizontalHeaderLabels([str(i) for i in range(self.table_widget.columnCount())])
        self.table_widget.setVerticalHeaderLabels([str(i) for i in range(self.table_widget.rowCount())])
        # This fills the table with QLineEdits, which are filled with the data from the ["sas"] array
        for column in range(self.table_widget.columnCount()):
            for row in range(self.table_widget.rowCount()):
                cellValue = QLineEdit()
                cellValue.setText(str(float(self.main_window.world['sas'][int(self.main_window.index)][column][row])))
                cellValue.setFrame(False)
                self.table_widget.setCellWidget(row, column, cellValue)
        # make save button
        button_apply = QPushButton()
        button_apply.setText('Apply Changes')
        button_apply.clicked.connect(self.apply_changes)
        # set layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.table_widget)
        self.layout.addWidget(button_apply)

    def apply_changes(self):
        '''
        This function closes the advanced settings window and applies the change made

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # Array to save the changed transition probabilities and check if the entered data is valid - smaller than 1
        transition_probabilities = np.zeros((4, self.main_window.world['states']))
        for column in range(self.table_widget.columnCount()):
            for row in range(self.table_widget.rowCount()):
                transition_probabilities[column][row] = float(self.table_widget.cellWidget(row, column).text())
        # ensure that probabilities sum to one
        transition_probabilities /= np.sum(transition_probabilities, axis=1).reshape((4, 1))
        for state in range(self.main_window.world['states']):
            for action in range(4):
                self.main_window.world['sas'][self.main_window.index][action][state] = transition_probabilities[action][state]        
        self.parent().close()


class NewDialog(QDialog):
    
    def __init__(self, parent: MainWindow):
        '''
        This class implements the Gridworld editor's new gridworld dialog.

        Parameters
        ----------
        parent :                            The new gridworld dialog's parent window.

        Returns
        ----------
        None
        '''
        super().__init__(parent)
        self.parent = parent
        # prepare dialog
        self.setWindowTitle('New Gridworld')
        self.layout = QVBoxLayout()
        self.QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(self.QBtn)
        self.button_box.accepted.connect(self.apply)
        self.button_box.rejected.connect(self.reject)
        # parameter input
        self.validator = QIntValidator(1, 100)
        self.label_width = QLabel('Set gridworld width:')
        self.input_width = QLineEdit()
        self.input_width.setPlaceholderText('Height/Rows(x)')
        self.input_width.setValidator(self.validator)
        self.label_height = QLabel('Set gridworld height:')
        self.input_height = QLineEdit()
        self.input_height.setPlaceholderText('Width/Columns(y)')
        self.input_height.setValidator(self.validator)
        self.label_state_type = QLabel('Set default state type:')
        self.radio_button_start = QRadioButton('Starting')
        self.radio_button_none = QRadioButton('None')
        self.radio_button_none.click()
        # add elements to panel
        self.layout.addWidget(self.label_width)
        self.layout.addWidget(self.input_width)
        self.layout.addWidget(self.label_height)
        self.layout.addWidget(self.input_height)
        self.layout.addWidget(self.label_state_type)
        self.layout.addWidget(self.radio_button_start)
        self.layout.addWidget(self.radio_button_none)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)
        
    def apply(self):
        '''
        This functions generates a new gridworld with the entered parameters and updates the GUI.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # make gridworld
        width, height = int(self.input_width.text()), int(self.input_height.text())
        world = gwt.make_gridworld(height, width, invalid_transitions=[])
        if not self.radio_button_start.isChecked():
            world['starting_states'] = np.array([], dtype=int)
        # update GUI
        self.parent.world = world
        self.parent.changed = False
        self.parent.file = None
        self.parent.index = 0
        self.parent.central_panel = CentralPanel(self.parent)
        self.parent.setCentralWidget(self.parent.central_panel)
        self.accept()
        

class TemplateDialog(QDialog):

    def __init__(self, parent: MainWindow, maze: str):
        '''
        This class implements the Gridworld editor's template dialog.
        According to the choice of template the dialog is generated dynamically.

        Parameters
        ----------
        parent :                            The template dialog's parent window.
        maze :                              The template type (i.e., \'t_maze\', \'double_t_maze\', \'two_sided_t_maze\', \'8_maze\', \'two_choice_t_maze\', \'detour_maze\').

        Returns
        ----------
        None
        '''
        super().__init__(parent)
        self.parent, self.maze = parent, maze
        # define template parameters
        self.templates = {'t_maze': {'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                                     'stem_length': {'type': int, 'range': [1, None], 'label': 'Stem Length'},
                                     'goal_arm': {'type': list, 'range': ['left', 'right'], 'label': 'Goal Arm'},
                                     'reward': {'type': float, 'range': [None, None], 'label': 'Reward'}},
                          'double_t_maze': {'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                                     'stem_length': {'type': int, 'range': [1, None], 'label': 'Stem Length'},
                                     'goal_arm': {'type': list, 'range': ['left-left', 'left-right', 'right-left', 'right-right'], 'label': 'Goal Arm'},
                                     'reward': {'type': float, 'range': [None, None], 'label': 'Reward'}},
                          'two_sided_t_maze': {'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                                     'stem_length': {'type': int, 'range': [1, None], 'label': 'Stem Length'},
                                     'goal_arm': {'type': list, 'range': ['left-left', 'left-right', 'right-left', 'right-right'], 'label': 'Goal Arm'},
                                     'reward': {'type': float, 'range': [None, None], 'label': 'Reward'}},
                          '8_maze': {'center_height': {'type': int, 'range': [1, None], 'label': 'Height'},
                                     'lap_width': {'type': int, 'range': [1, None], 'label': 'Lap Width'},
                                     'goal_location': {'type': list, 'range': ['left', 'right'], 'label': 'Goal Location'},
                                     'reward': {'type': float, 'range': [None, None], 'label': 'Reward'}},
                          'two_choice_t_maze': {'center_height': {'type': int, 'range': [1, None], 'label': 'Height'},
                                     'lap_width': {'type': int, 'range': [1, None], 'label': 'Lap Width'},
                                     'arm_length': {'type': int, 'range': [1, None], 'label': 'Arm Length'},
                                     'chirality': {'type': list, 'range': ['left', 'right'], 'label': 'Chirality'},
                                     'goal_location': {'type': list, 'range': ['left', 'right'], 'label': 'Goal Location'},
                                     'reward': {'type': float, 'range': [None, None], 'label': 'Reward'}},
                          'detour_maze': {'width_small': {'type': int, 'range': [1, None], 'label': 'Width Small'},
                                     'height_small': {'type': int, 'range': [1, None], 'label': 'Height Small'},
                                     'width_large': {'type': int, 'range': [1, None], 'label': 'Width Large'},
                                     'height_large': {'type': int, 'range': [1, None], 'label': 'Height Large'},
                                     'reward': {'type': float, 'range': [None, None], 'label': 'Reward'}},}
        # define template labels
        self.template_labels = {'t_maze' : 'T-maze', 'double_t_maze' : 'Double T-maze', 'two_sided_t_maze' : 'Two-sided T-maze',
                                '8_maze' : '8-maze', 'two_choice_t_maze' : 'Two-choice T-maze', 'detour_maze' : 'Detour Maze'}
        # register functions
        self.template_functions = {'t_maze' : gwt.make_t_maze, 'double_t_maze' : gwt.make_double_t_maze,
                                   'two_sided_t_maze' : gwt.make_two_sided_t_maze, '8_maze' : gwt.make_8_maze,
                                   'two_choice_t_maze' : gwt.make_two_choice_t_maze, 'detour_maze' : gwt.make_detour_maze}
        # prepare dialog
        self.setWindowTitle('Configure %s template!' % self.template_labels[self.maze])
        self.layout = QVBoxLayout()
        self.param_layout = QVBoxLayout()
        self.QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(self.QBtn)
        self.button_box.accepted.connect(self.apply)
        self.button_box.rejected.connect(self.reject)
        # generate template specific layout
        self.labels, self.params = {}, {}
        for param in self.templates[self.maze]:
            # param label
            self.labels[param] = QLabel(self.templates[self.maze][param]['label'])
            # param input
            if self.templates[self.maze][param]['type'] in [int, float]:
                self.params[param] = QLineEdit()
            elif self.templates[self.maze][param]['type'] == list:
                self.params[param] = QComboBox()
                self.params[param].addItems(self.templates[self.maze][param]['range'])
            # add to layout
            self.param_layout.addWidget(self.labels[param])
            self.param_layout.addWidget(self.params[param])
        self.layout.addLayout(self.param_layout)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)
        
    def apply(self):
        '''
        This functions generates a new gridworld from a selected template and updates the GUI.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # retrieve parameters
        params = {}
        for param in self.params:
            if self.templates[self.maze][param]['type'] in [int, float]:
                params[param] = self.templates[self.maze][param]['type'](self.params[param].text())
            else:
                params[param] = self.params[param].currentText()
        # generate gridworld from template
        world = self.template_functions[self.maze](**params)
        # invalid parameters yield an empty dictionary
        if len(world) == 0:
            QMessageBox.critical(self, 'Invalid Parameters!', 'Please adjust template parameters.')
            return
        world['starting_states'] = np.array([]).astype(int)
        # update GUI
        self.parent.world = world
        self.parent.changed = False
        self.parent.file = None
        self.parent.index = 0
        self.parent.central_panel = CentralPanel(self.parent)
        self.parent.setCentralWidget(self.parent.central_panel)
        self.accept()


class UnsavedChangesDialog(QDialog):

    def __init__(self, parent: MainWindow):
        '''
        The gridworld editor's unsaved changes dialog class.

        Parameters
        ----------
        parent :                            The unsaved changes dialog's parent window.

        Returns
        ----------
        None
        '''
        super().__init__(parent)
        self.parent = parent
        self.layout = QGridLayout()
        self.setWindowTitle('Unsaved Changes')
        # prepare elements
        self.label_dialog = QLabel()
        self.label_dialog.setText('The file has been modified. Do you want to save the changes?')
        self.button_yes = QPushButton('Yes')
        self.button_no = QPushButton('No')
        self.button_cancel = QPushButton('Cancel')
        # make layout
        self.layout.addWidget(self.label_dialog, 0, 0, 1, 3)
        self.layout.addWidget(self.button_yes, 1, 0, 1, 1)
        self.layout.addWidget(self.button_no, 1, 1, 1, 1)
        self.layout.addWidget(self.button_cancel, 1, 2, 1, 1)
        self.button_yes.clicked.connect(self.save_current_file)
        self.button_no.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)
        self.setLayout(self.layout)

    def save_current_file(self):
        '''
        This function saves the gridworld in its currently selected file.
        If none is currently selected the user will be asked to choose one.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        self.hide()
        self.parent.save()
        if self.changed == False:
            self.accept()
        else:
            self.reject()


class InfoDialog(QDialog):

    def __init__(self, parent: MainWindow):
        '''
        The gridworld editor's info dialog class.

        Parameters
        ----------
        parent :                            The info dialog's parent window.

        Returns
        ----------
        None
        '''
        super().__init__(parent)
        self.parent = parent
        self.layout = QGridLayout()
        self.setWindowTitle('Info')
        self.label_dev = QLabel('Developed by : William Forchap, Kilian Kandt, Marius Tenhumberg, Chuan Jin, Umut Yilmaz, Nicolas Diekmann')
        self.label_sup = QLabel('Supervised by : Nicolas Diekmann')
        self.button_ok = QPushButton('Ok')
        self.layout.addWidget(self.label_dev, 0, 0, 1, 1)
        self.layout.addWidget(self.label_sup, 1, 0, 1, 1)
        self.layout.addWidget(self.button_ok, 2, 0, 1, 1)
        self.button_ok.clicked.connect(self.accept)
        self.setLayout(self.layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = MainWindow()
    a.show()
    app.exec()
