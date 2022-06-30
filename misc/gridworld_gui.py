# basic imports
import sys
import math
import pickle
import numpy as np
# framework imports
import gridworld_tools as gridTools
#Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QBrush, QColor, QIntValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QAction, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, \
    QLabel, QRadioButton, QTableWidget, QGraphicsScene, QLineEdit, QGraphicsView, QSplitter, QPushButton, QFileDialog
from PyQt5 import QtWidgets


def updateProbability(world, index, probabilities=[]):
    '''
    Updates the world dictionary according to the applied changes in the GUI advanced settings table.
    Takes the index to determine at what position in the directory the changes have to be made.

    | **Args**
    | world:                        The gridworld dictionary created using the  makeGridworld function
    | index:                        The index of the state that is being edited
    | probabilities:                The changed transistion probabilities
    '''
    for state in range(world['states']):
        for action in range(4):
            world['sas'][index][action][state] = probabilities[action][state]

def updateTransitions(world, invalidStates=[], invalidTransitions=[]):
    '''
    Updates the world dictionary according to the applied changes, through douuble clicking the state borders.

    | **Args**
    | world:                        The gridworld dictionary created using the makeGridworld function
    | invalidStates:                All states which have become unreachable
    | invalidTransitions:           All transistion which have turned invalid
    '''
    world['invalidStates'] = invalidStates
    world['invalidTransitions'] = invalidTransitions
    #Updates the state-action-state array according to the now invalid states
    world['sas'] = np.zeros((world['states'], 4, world['states']))
    for state in range(world['states']):
        for action in range(4):
            h = int(state / world['width'])
            w = state - h * world['width']
            # left
            if action == 0:
                w = max(0, w - 1)
            # up
            elif action == 1:
                h = max(0, h - 1)
            # right
            elif action == 2:
                w = min(world['width'] - 1, w + 1)
            # down
            else:
                h = min(world['height'] - 1, h + 1)
            # apply wind
            # currently walls are not taken into account!
            h += world['wind'][state][0]
            w += world['wind'][state][1]
            h = min(max(0, h), world['height'] - 1)
            w = min(max(0, w), world['width'] - 1)
            # determine next state
            nextState = int(h * world['width'] + w)
            if nextState in world['invalidStates'] or (state, nextState) in world['invalidTransitions']:
                nextState = state
            world['sas'][state][action][nextState] = 1

def updateState(world, index, reward, terminal, starting):
    '''
    Updates the world dictionary according to the applied changes in the GUI state information side panel.
    Takes the index to determine at what position in the diretory the changes have to be made.

    | **Args**
    | world:                        The gridworld dictionary created using the  makeGridworld function
    | index:                        The index of the state that is being edited
    | reward:                       The reward value that will be used for the world directory
    | terminal:                     Is true if the state will be of type terminal
    | starting:                     Is true if the state willbe of type starting
    '''
    # update terminal status
    world['terminals'][index] = int(terminal)
    #removes index from the starting state list
    if not starting:
        world['startingStates'] = world['startingStates'][world['startingStates'] != index]
    #add index as new starting state if it does not already exist
    elif starting and not(index in world['startingStates']):
        world['startingStates'] = np.append(world['startingStates'], index)
    # update reward value
    world['rewards'][index] = reward


class Settings():
    WORLD = dict()
    # Border size and cell size
    WIDTH = 50
    HEIGHT = 50
    BORDER_WIDTH = 5
    index = 0
    InvalidTransistions = []
    InvalidStates = []
    changed = False
    file = None


class StartWindowPanel(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QVBoxLayout()
        # set up height and width input fields
        self.validator = QIntValidator(0, 2147483647)
        self.field_height = QLineEdit()
        self.field_height.setText('Height/Rows(x)')
        self.field_height.setValidator(self.validator)
        self.field_width = QLineEdit()
        self.field_width.setText('Width/Collumns(y)')
        self.field_width.setValidator(self.validator)
        # prepare starting state setting selection
        self.label_state_type = QLabel('Set default state type:')
        self.radio_button_start = QRadioButton('Starting')
        self.radio_button_start.click()
        self.radio_button_none = QRadioButton('None')
        # prepare buttons for gridworld loading/generation
        self.button_generate = QPushButton('Generate Gridworld')
        self.button_load = QPushButton('Load Gridworld')
        # add elements to panel
        self.layout.addWidget(self.field_height)
        self.layout.addWidget(self.field_width)
        self.layout.addWidget(self.button_generate)
        self.layout.addWidget(self.button_load)
        self.layout.addWidget(self.label_state_type)
        self.layout.addWidget(self.radio_button_start)
        self.layout.addWidget(self.radio_button_none)
        self.setLayout(self.layout)


class StartWindow(QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QMainWindow.__init__(self)
        self.setWindowTitle('Start')
        self.start_widget = StartWindowPanel()
        self.setCentralWidget(self.start_widget)
        self.start_widget.button_generate.clicked.connect(self.clicked_generate)
        self.start_widget.button_load.clicked.connect(self.clicked_load)
        self.resize(250, 200)

    def clicked_generate(self):
        '''
        This function takes the properties set by the user and generates the gridworld accordingly.
        '''
        # if statement makes sure you can't start with the inital text in the input fields
        if (self.start_widget.field_width.text().isnumeric() and self.start_widget.field_height.text().isnumeric()):
            # console test text
            print(self.start_widget.field_height.text(), 'x', self.start_widget.field_width.text())
            # create gridworld
            Settings.WORLD = gridTools.makeEmptyField(int(self.start_widget.field_height.text()),
                                                      int(self.start_widget.field_width.text()))
            if not self.start_widget.radio_button_start.isChecked():
                Settings.WORLD['startingStates'] = np.array([]).astype(int)
            # open main window, close this window
            self.main_window = MainWindow()
            self.main_window.show()
            self.close()
        else:
            print("please enter width and height values")

    def clicked_load(self):
        '''
        This function lets the user open a *.pkl file to load a previously saved gridworld.
        '''
        file_name = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', 'Pickle files (*.pkl)')[0]
        print('Try loading file: ', file_name)
        try:
            Settings.changed = False
            Settings.file = file_name
            Settings.WORLD = pickle.load(open(file_name, 'rb'))
            Settings.InvalidTransistions = Settings.WORLD['invalidTransitions']
            Settings.InvalidStates = Settings.WORLD['invalidStates']
            self.main_window = MainWindow()
            self.main_window.show()
            self.close()
            print('Successfully load file: ', file_name)
        except:
            print('Couldn\'t load file.')


class Line(QtWidgets.QGraphicsLineItem):
    '''
    An extended version of the standard QGraphicsLineItem which
    toggles the line color between gray and red on double click.
    
    | **Args**
    | x1:                           X-Coordinate of first point of line
    | y1:                           Y-Coordinate of first point of line
    | x2:                           X-Coordinate of last point of line
    | y2:                           Y-Coordinate of last point of line
    | pen:                          The pen, that draws the line
    '''
    def __init__(self, x1, y1, x2, y2, pen):
        super(Line, self).__init__()
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
        self.highlighted = False
        # determine the transition associated with this line
        self.transition = (0, 0)
        # vertical line
        if x1 == x2:
            coordX = y1 // Settings.HEIGHT
            coordY_left = (x1 - Settings.BORDER_WIDTH) // Settings.WIDTH
            coordY_right = (x1 + Settings.BORDER_WIDTH) // Settings.WIDTH
            self.transition = (int((coordX * Settings.WORLD['width']) + coordY_left),
                               int((coordX * Settings.WORLD['width']) + coordY_right))
        # horizontal line
        else:
            coordX_up = (y1 - Settings.BORDER_WIDTH) // Settings.HEIGHT
            coordX_down = (y1 + Settings.BORDER_WIDTH) // Settings.HEIGHT
            coordY = x1 // Settings.WIDTH
            self.transition = (int((coordX_up * Settings.WORLD['width']) + coordY),
                               int((coordX_down * Settings.WORLD['width']) + coordY))
            
    def mousePressEvent(self, event):
        '''
        This function checks if a border  has been double clicked. Un/highlights the border  and
        edits the WORLD dict accordingly

        | **Args**
        | event:                    The mouse press event.
        '''
        Settings.changed = True
        if not self.highlighted:
            self.setPen(QPen(Qt.red, Settings.BORDER_WIDTH))
            # add invalid transitions
            Settings.InvalidTransistions.append(self.transition)
            Settings.InvalidTransistions.append(self.transition[::-1])
        else:
            self.setPen(QPen(Qt.gray, Settings.BORDER_WIDTH))
            # remove invalid transitions
            Settings.InvalidTransistions.pop(Settings.InvalidTransistions.index(self.transition))
            Settings.InvalidTransistions.pop(Settings.InvalidTransistions.index(self.transition[::-1]))
        # update the gridworld dictionary
        updateTransitions(Settings.WORLD, Settings.InvalidStates, Settings.InvalidTransistions)
        self.highlighted = not self.highlighted
        

class Grid(QGraphicsScene):
    '''
    This class represents the grid of the gridworld.
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        QGraphicsView.__init__(self)
        self.parent = parent
        self.lines = []
        self.highlight = None
        self.height = Settings.WORLD['height'] * Settings.HEIGHT
        self.width = Settings.WORLD['width'] * Settings.WIDTH
        self.pen = QPen(QColor(125, 175, 240, 125), 0)
        self.brush = QBrush(QColor(125, 175, 240, 125))
        self.draw_grid()
        self.symbols_terminal = np.empty((Settings.WORLD['states'], 1), QGraphicsScene)
        self.symbols_starting = np.empty((Settings.WORLD['states'], 1), QGraphicsScene)

    def draw_grid(self):
        '''
        This function draws the grid.
        '''
        pen_lines = QPen(Qt.gray, Settings.BORDER_WIDTH)
        pen_border = QPen(Qt.black, Settings.BORDER_WIDTH)
        # lines parallel to GUI x-axis (vertical)
        for column in range(1, Settings.WORLD['width']):
            x = column * Settings.WIDTH
            for row in range(0, Settings.WORLD['height']):
                y = row * Settings.HEIGHT
                line = Line(x, y, x, y + Settings.HEIGHT, pen_lines)
                if line.transition in Settings.WORLD['invalidTransitions']:
                    line.highlight()
                self.lines.append(self.addItem(line))
        # lines parallel to GUI y-axis (horizontal)
        for row in range(1, Settings.WORLD['height']):
            y = row * Settings.HEIGHT
            for column in range(0, Settings.WORLD['width']):
                x = column * Settings.WIDTH
                line = Line(x, y, x + Settings.WIDTH, y, pen_lines)
                if line.transition in Settings.WORLD['invalidTransitions']:
                    line.highlight()
                self.lines.append(self.addItem(line))
        # Outer border
        self.lines.append(self.addLine(0, 0, self.width, 0, pen_border))
        self.lines.append(self.addLine(0, 0, 0, self.height, pen_border))
        self.lines.append(self.addLine(0, self.height, self.width, self.height, pen_border))
        self.lines.append(self.addLine(self.width, 0, self.width, self.height, pen_border))

    def set_visible(self, visible=True):
        '''
        Sets the grid visible/invisible.

        | **Args**
        | visible:                  Boolean, whether the grid is visible or not.
        '''
        for line in self.lines:
            line.setVisible(visible)

    def delete_grid(self):
        '''
        Deletes grid
        '''
        for line in self.lines:
            self.removeItem(line)
        del self.lines[:]

    def set_opacity(self, opacity):
        '''
        This function sets the opacity of the lines.

        | **Args**
        | opacity:                  The opacity the lines will have.
        '''
        for line in self.lines:
            line.setOpacity(opacity)

    def highlight_state(self, x, y):
        '''
        This function highlights a state with a blue square.

        | **Args**
        | x:                        State's x-coordinate.
        | y:                        State's y-coordinate.

        '''
        # removes pieces old highlight
        self.removeItem(self.highlight)
        self.highlight = self.addRect(y * Settings.WIDTH + Settings.BORDER_WIDTH / 2,
                                      x * Settings.HEIGHT + Settings.BORDER_WIDTH / 2,
                                      Settings.WIDTH - Settings.BORDER_WIDTH,
                                      Settings.HEIGHT - Settings.BORDER_WIDTH,
                                      self.pen, self.brush)
        # removes artefacts of old highlights
        self.update()

    def highlight_terminal_starting(self, terminals, startings):
        '''
        This function highlights the states according to thet erminal and
        starting state list in the world dictionary.

        | **Args**
        | terminals:                The terminal state list in the WORLD dict.
        | startings:                The starting state  list in the WOLRD dict.
        '''
        # delete all symbols created before and then empty the array
        for index in range(len(self.symbols_terminal)):
            self.removeItem(self.symbols_terminal[int(index), 0])
        for index in range(len(self.symbols_starting)):
            self.removeItem(self.symbols_starting[int(index), 0])
        # create symbol at position x,y for all terminals and startings
        for index in np.arange(terminals.shape[0])[terminals == 1]:
            # calculate position
            y = (index // Settings.WORLD['width']) * Settings.WIDTH
            x = (index % Settings.WORLD['width']) * Settings.HEIGHT
            # place X at position (x,y)
            self.symbols_terminal[int(index), 0] = self.addText('X')
            self.symbols_terminal[int(index), 0].setPos(x, y)
        for index in startings:
            # calculate position
            y = (index // Settings.WORLD['width']) * Settings.WIDTH
            x = (index % Settings.WORLD['width']) * Settings.HEIGHT
            # place S at position (x,y)
            self.symbols_starting[int(index), 0] = self.addText('S')
            self.symbols_starting[int(index), 0].setPos(x, y)
        self.update()

    def mousePressEvent(self, event):
        '''
        This function determines which state or if a border has been clicked on.
        
        | **Args**
        | event:                       The triggering event.
        '''
        # Gets mouse position within scene
        posX, posY = event.scenePos().x(), event.scenePos().y()
        # check if mouse within scene at all
        if posX >= 0 and posY >= 0 and posX <= self.width and posY <= self.height:
            # check if mouse position is on border (rel = relative pos in state)
            relX = (posX + Settings.BORDER_WIDTH / 2) % Settings.WIDTH
            relY = (posY + Settings.BORDER_WIDTH / 2) % Settings.HEIGHT
            if relX > Settings.BORDER_WIDTH + 0.5 and relY > Settings.BORDER_WIDTH + 0.5:
                # translate to coordinate
                coordX, coordY = int(posY // Settings.HEIGHT), int(posX // Settings.WIDTH)
                # calc index
                index = coordX * Settings.WORLD['width'] + coordY
                # Console output of state coordinates
                coordText = '(' + str(coordX) + ',' + str(coordY) + ')'
                # highlight current state
                self.highlight_state(coordX, coordY)
                # update state info
                self.parent.change_state_information(coordText, index)


class GridViewer(QGraphicsView):
    '''
    The grid view class.
    
    | **Args**
    | scene:                        The grid scene.
    | parent:                       The parent window.
    '''
    def __init__(self, scene, parent=None):
        super().__init__(parent)
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

    def wheelEvent(self, event):
        '''
        This function scales the view depending on the wheel event.
        
        | **Args**
        | event:                       The triggering event.
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
            self.fitInView(self.scene.itemsBoundingRect(),Qt.KeepAspectRatio)
        
        self.scene.update()
            
    def showEvent(self, event):
        '''
        This function fits the scene in view as soon as the view is shown.
        
        | **Args**
        | event:                       The triggering event.
        '''
        super().showEvent(event)
        self.fitInView(self.scene.itemsBoundingRect(),Qt.KeepAspectRatio)
        # Determines maximum zoom count so you cannot zoom further when one
        # state fits the view
        maxFactor = max(self.scene.itemsBoundingRect().width() /self.viewport().rect().width(),
                        self.scene.itemsBoundingRect().height() / self.viewport().rect().height())
        maxFactor *= min(self.viewport().rect().width()/Settings.WIDTH,
                         self.viewport().rect().height()/Settings.HEIGHT)
        if maxFactor > 1:
            self.max_zoom = math.floor(math.log(maxFactor, 5/4))


class StateInformation(QWidget):
    '''
    The state information class.
    
    | **Args**
    | parent:                       The parent window.
    '''
    def __init__(self, parent=None):
        super(StateInformation, self).__init__(parent=parent)
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

    def change_state_information(self, text_coordinates, index):
        '''
        this function is used  to  update the state information panel
        text according to the selected state.

        | **Args**
        | textCoord: The  states coordinates
        | index: The states index
        '''
        self.label_index.setText('Index: ' + str(int(index)))
        self.index = int(index)
        self.label_coordinates.setText('Coordinates: ' + text_coordinates)
        self.field_reward.setText(str(float(Settings.WORLD['rewards'][int(index)])))
        if Settings.WORLD['terminals'][int(index)]:
            self.radio_button_start.click()
        elif index in Settings.WORLD['startingStates']:
            self.radio_button_start.click()
        else:
            self.radio_button_none.click()

    def update_state(self):
        '''
        This function is used to update the selected states properties inside
        the WORLD dictionary.
        '''
        Settings.changed = True
        updateState(Settings.WORLD, int(self.index), float(self.field_reward.text()),
                    self.radio_button_terminal.isChecked(), self.radio_button_start.isChecked())
        self.parent().parent().highlight_terminal_starting(Settings.WORLD["terminals"],
                                                           Settings.WORLD["startingStates"])

    def open_advanced(self):
        '''
        This function opens the advanced settings menu.
        '''
        Settings.index = int(self.index)
        self.advanced_settings = AdvancedSettingsWindow()
        self.advanced_settings.show()


class AdvancedSettingsWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QMainWindow.__init__(self)
        self.setWindowTitle('Advanced Settings')
        self.resize(500, 500)
        self.advanced_widget = AdvancedSettingsWidget()
        self.setCentralWidget(self.advanced_widget)


class AdvancedSettingsWidget(QWidget):
    '''
    The Widget, that handles the table for the transition probabilities.
    '''
    def __init__(self):
        QWidget.__init__(self)
        # The table with 4 columns and as many roows as there are states
        self.table_widget = QTableWidget(Settings.WORLD['states'], 4, self)
        self.table_widget.setHorizontalHeaderLabels([str(i) for i in range(self.table_widget.columnCount())])
        self.table_widget.setVerticalHeaderLabels([str(i) for i in range(self.table_widget.rowCount())])
        # This fills the table with QLineEdits, which are filled with the data from the ["sas"] array
        for column in range(self.table_widget.columnCount()):
            for row in range(self.table_widget.rowCount()):
                cellValue = QLineEdit()
                cellValue.setText(str(float(Settings.WORLD['sas'][int(Settings.index)][column][row])))
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
        This function closes the advanced settings window and applies the changes made.
        '''
        #Array to save the changed transition probabilities and check if the entered data is valid - smaller than 1
        transition_probabilities = np.zeros((4, Settings.WORLD['states']))
        for column in range(self.table_widget.columnCount()):
            for row in range(self.table_widget.rowCount()):
                transition_probabilities[column][row] = float(self.table_widget.cellWidget(row, column).text())
        # ensure that probabilities sum to one
        transition_probabilities /= np.sum(transition_probabilities, axis=1).reshape((4, 1))
        updateProbability(Settings.WORLD, Settings.index, transition_probabilities)
        self.parent().close()


class CentralPanel(QWidget):
    '''
    The Widget, that handles the two main widgets (grid and state information).
    '''
    def __init__(self):
        QWidget.__init__(self)
        # Grid and View
        self.scene = Grid(self)
        self.view = GridViewer(self.scene, self)
        self.scene.highlight_terminal_starting(Settings.WORLD["terminals"], Settings.WORLD["startingStates"])
        # Sidebar
        self.state_menu = StateInformation(self)
        # splitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.state_menu)  # sidebar here
        self.splitter.setStretchFactor(0, 10)
        # add splitter(Grid & InfoBar) to layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)

    def change_state_information(self, textCoord, index):
        '''
        Used to pass the fucntion to the children of Central Panel.

        | **Args**
        | textCoord:  states coordinates
        | index: states index
        '''
        self.state_menu.change_state_information(textCoord, index)

    def highlight_terminal_starting(self, terminals, startings):
        '''
        Used to pass the function to the state info menu

        | **Args**
        | terminals: worlds terminal states
        | startings: worlds starting states
        '''
        self.scene.highlight_terminal_starting(terminals, startings)


class MainWindow(QMainWindow):
    '''
    This class implements the main window.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QMainWindow.__init__(self)
        self.setWindowTitle('Gridworld Editor')
        self.create_MenuBar()
        self.central_panel = CentralPanel()
        self.setCentralWidget(self.central_panel)
        # Window size
        self.resize(1000, 563)

    def create_MenuBar(self):
        '''
        This function creates the menu bar.
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
        # info
        self.action_info = QAction('&Info', self)
        self.action_info.setShortcut("Ctrl+I")
        self.action_info.triggered.connect(self.info)
        # Create menuBar
        self.menu_bar = self.menuBar()
        # Add menus to menuBar
        self.file_menu = self.menu_bar.addMenu('&File')
        self.info_menu = self.menu_bar.addMenu('&Info')
        # Add actions to menu
        self.file_menu.addAction(self.action_new)
        self.file_menu.addAction(self.action_open)
        self.file_menu.addAction(self.action_save)
        self.file_menu.addAction(self.action_save_as)
        self.file_menu.addAction(self.action_quit)
        self.info_menu.addAction(self.action_info)

    def new(self):
        '''
        This function returns the user to the starting window.
        '''
        if not self.unsaved_changes():
            Settings.changed = False
            Settings.file = None
            Settings.InvalidTransistions = []
            Settings.InvalidStates = []
            self.start_widget = StartWindow()
            self.start_widget.show()
            self.close()

    def load(self):
        '''
        This function asks the user to select a file and opens it.
        '''
        if not self.unsaved_changes():
            file_name = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Pickle files (*.pkl)")[0]
            print('Try loading file: ', file_name)
            try:
                Settings.changed = file_name
                Settings.file = file_name
                Settings.WORLD = pickle.load(open(file_name, 'rb'))
                Settings.InvalidTransistions = Settings.WORLD['invalidTransitions']
                Settings.InvalidStates = Settings.WORLD['invalidStates']
                self.main_window = MainWindow()
                self.main_window.show()
                self.close()
                print('Successfully loaded file. ')
            except:
                print('Couldn\'t load file.')

    def save_as(self):
        '''
        This function saves the current gridworld in a file
        
        | **Args**
        | file_name:                    File name/Target file.
        '''
        file_name = QFileDialog.getSaveFileName(self, 'Open file', 'c:\\', "Pickle files (*.pkl)")[0]
        print('Try saving file: ', file_name)
        try:
            pickle.dump(Settings.WORLD, open(file_name, 'wb'))
            Settings.changed = False
            Settings.file = file_name
            print('Successfully saved file. ')
        except:
            print('Couldn\'t save file.')

    def save(self):
        '''
        This function asks the user to select a file save the gridworld in it.
        '''
        if Settings.file is None:
            self.save_as()
        else:
            print('Try saving file: ', Settings.file)
            try:
                pickle.dump(Settings.WORLD, open(Settings.file, 'wb'))
                Settings.changed = False
                print('Successfully saved file. ')
            except:
                print('Couldn\'t save file.')

    def unsaved_changes(self):
        '''
        This function opens a dialog window to asks the user to save the
        gridworld if there are unsaved changes.
        '''
        # Returns False if there are no unsaved changes/if these should be ignored
        if Settings.changed:
            print('Unsaved changes.')
            self.unsaved_changes_dialog = unsavedChangesDialog(self)
            if self.unsaved_changes_dialog.exec():
                return False
            else:
                return True

        return False
    
    def info(self):
        '''
        This function opens the info dialog
        '''
        self.info_dialog = infoDialog(self)
        self.info_dialog.exec()

    def closeEvent(self, event):
        '''
        This function closes the programm.
        '''
        if not self.unsaved_changes():
            super().closeEvent(event)
        else:
            event.ignore()

class unsavedChangesDialog(QDialog):
    '''
    This class implements an unsaved changes dialog.

    | **Args**
    | parent:                       The parent widget.
    '''
    def __init__(self, parent=None):
        super(unsavedChangesDialog, self).__init__(parent)
        QDialog.__init__(self)
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
        If no file is currently selected the user is asked to choose one.
        '''
        self.hide()
        self.parent.save()
        if Settings.changed == False:
            self.accept()
        else:
            self.reject()
            
class infoDialog(QDialog):
    '''
    This class implements an info dialog.

    | **Args**
    | parent:                       The parent widget.
    '''
    def __init__(self, parent=None):
        super(infoDialog, self).__init__(parent)
        QDialog.__init__(self)
        self.parent = parent
        self.layout = QGridLayout()
        self.setWindowTitle('Info')
        self.label_dev = QLabel('Developed by : William Forchap, Kilian Kandt, Marius Tenhumberg')
        self.label_sup = QLabel('Supervised by : Nicolas Diekmann')
        self.button_ok = QPushButton('Ok')
        self.layout.addWidget(self.label_dev, 0, 0, 1, 1)
        self.layout.addWidget(self.label_sup, 1, 0, 1, 1)
        self.layout.addWidget(self.button_ok, 2, 0, 1, 1)
        self.button_ok.clicked.connect(self.accept)
        self.setLayout(self.layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = StartWindow()
    a.show()
    app.exec()
