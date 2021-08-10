# basic imports
import sys
import math
import pickle
import numpy as np
# framework imports
import gridworld_tools
import gridworld_tools as gridTools
#Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QBrush, QColor, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QAction, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, \
    QLabel, QRadioButton, QTableWidget, QGraphicsScene, QLineEdit, QGraphicsView, QSplitter, QPushButton, QFileDialog
from PyQt5 import QtWidgets


def updateProbability(world, index, probabilities=[]):
    '''
    Updates the world  directory according to the applied changes in the GUI advanced settings table.
    Takes the index to determine at what position in the directory the changes have to be made.

    | **Args**
    | world: The gridworld  directory created using the  makeGridworld function
    | index: The index of the state that is being edited
    | probabilities: The changed transistion probabilities
    '''
    for state in range(world['states']):
        for action in range(4):
            world["sas"][index][action][state] = probabilities[action][state]

# Funktion um invalid States und invalid Transitions zu bearbeiten
def updateTransitions(world, invalidStates=[], invalidTransitions=[]):
    '''
    Updates the world  directory according to the applied changes, through douuble clicking the state borders.

    | **Args**
    | world: The gridworld directory created using the makeGridworld function
    | invalidStates: All states which have become unreachable
    | invalidTransitions: All transistion which have turned invalid
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

def updateState(world, index, reward, terminal, starting, none):
    '''
    Updates the world  directory according to the applied changes in the GUI state information side panel.
    Takes the index to determine at what position in the diretory the changes have to be made.

    | **Args**
    | world: The gridworld  directory created using the  makeGridworld function
    | index: The index of the state that is being edited
    | reward: The reward value that will be used for the world directory
    | terminal: Is true if the state will be of type terminal
    | starting: Is true if the state willbe of type starting
    | none: Is true if the state will be neither terminal nor starting
    '''
    #add new terminal state at position index to the directory
    if terminal:
        world["terminals"][index] = 1
    #remove terminal state
    elif not terminal:
        world["terminals"][index] = 0
    #removes index from the starting state list
    if not starting:
        world["startingStates"] = world["startingStates"][world["startingStates"] != index]
    #add index as new starting state if it does not already exist
    elif(starting and not(index in world["startingStates"])):
        world["startingStates"] = np.append(world["startingStates"], index)
    #add new reward to world.["rewards"]. change value for the state index
    world["rewards"][index] = reward

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
    file = ""


class StartWindowPanel(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QVBoxLayout()

        # makes sure you can't type in strings etc.
        validator = QIntValidator(0, 2147483647)
        self.heightWidget = QLineEdit()
        self.heightWidget.setText("Height/Rows(x)")
        self.heightWidget.setValidator(validator)
        self.widthWidget = QLineEdit()
        self.widthWidget.setText("Width/Collumns(y)")
        self.widthWidget.setValidator(validator)

        self.radioLabel = QLabel("Set default state type:")
        self.startingBtn = QRadioButton("Starting")
        self.startingBtn.click()
        self.noneBtn = QRadioButton("None")

        self.layout.addWidget(self.heightWidget)
        self.layout.addWidget(self.widthWidget)

        self.b1 = QPushButton("Generate Gridworld")
        self.b2 = QPushButton("Load Gridworld")
        self.layout.addWidget(self.b1)
        self.layout.addWidget(self.b2)
        self.layout.addWidget(self.radioLabel)
        self.layout.addWidget(self.startingBtn)
        self.layout.addWidget(self.noneBtn)
        self.setLayout(self.layout)


class StartWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QMainWindow.__init__(self)
        
        self.setWindowTitle("Start")
        self.StartWidget = StartWindowPanel()
        self.setCentralWidget(self.StartWidget)
        self.StartWidget.b1.clicked.connect(self.clickedGenerate)
        self.StartWidget.b2.clicked.connect(self.clickedLoad)
        self.resize(250, 200)

    def clickedGenerate(self):
        '''
        This function takes the properties set by the user and
        generates the gridworld accordingly
        '''
        # if statement to check if QLineEdits are positive
        width = self.StartWidget.widthWidget.text()
        height = self.StartWidget.heightWidget.text()
        # if statement makes sure you can't start with the inital text in the input fields
        if (width.isnumeric() and height.isnumeric()):
            # set width height from input
            self.StartWidget.height = self.StartWidget.heightWidget.text()
            self.StartWidget.width = self.StartWidget.widthWidget.text()
            # set starting or none starting by default
            # starting = self.startWidget.startingBtn.isChecked()
            starting = self.StartWidget.startingBtn.isChecked()
            # console test text
            print(self.StartWidget.height + " x " + self.StartWidget.width)
            # create gridworld from gridworld_tools.py
            Settings.WORLD = gridTools.makeEmptyField(int(height), int(width))
            if not starting:
                Settings.WORLD["startingStates"] = np.empty(0)

            # open main window, close this window
            self.b = MainWindow()
            self.b.show()
            self.close()
        else:
            print("please enter width and height values")

    def clickedLoad(self):
        '''
        This function lets the user open a *.pkl file to load a
        previously saved gridworld.
        '''
        print("LOAD")
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "Pickle files (*.pkl)")
        fname = fname[0]
        print(fname)
        if fname != "":
            Settings.changed = False
            Settings.file = fname
            Settings.WORLD = pickle.load(open(fname, 'rb'))

            Settings.InvalidTransistions = Settings.WORLD["invalidTransitions"]
            Settings.InvalidStates = Settings.WORLD["invalidStates"]
            self.b = MainWindow()
            self.b.show()
            self.close()
            print("LOAD SUCCESS")
        else:
            print("LOAD FAIL")


class Line(QtWidgets.QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, pen):
        '''
            An extended verion of the standard QGraphicsLineItem, to provide the on-click ability to change borders to invalid and turn
            red
            | **Args**
            | x1: X-Coordinate of first point of line
            | y1: Y-Coordinate of first point of line
            | x2: X-Coordinate of last point of line
            | y2: Y-Coordinate of last point of line
            | pen: The pen, that draws the line
        '''
        super(Line, self).__init__()
        self.setLine(x1, y1, x2, y2)
        self.setPen(pen)
        self.changed = False
        # Orientation vertical if false and horizontal if true
        self.orientation = False if x1 == x2 else True
        #Used to save position of item in array in order to pop them when they are no longer invalid
        self.indexList1 = 0
        self.indexList2 = 0
        # Saves the indexes of the adjacent states of the border
        self.index1 = 0
        self.index2 = 0

        if (self.orientation == False):
            coordX_left = y1 // Settings.HEIGHT
            coordY_left = (x1 - Settings.BORDER_WIDTH) // Settings.WIDTH
            coordX_right = y1 // Settings.HEIGHT
            coordY_right = (x1 + Settings.BORDER_WIDTH) // Settings.WIDTH
            # calc index
            self.index1 = int((coordX_left * Settings.WORLD['width']) + coordY_left)
            self.index2 = int((coordX_right * Settings.WORLD['width']) + coordY_right)
        else:
            coordX_up = (y1 - Settings.BORDER_WIDTH) // Settings.HEIGHT
            coordY_up = x1 // Settings.WIDTH
            coordX_down = (y1 + Settings.BORDER_WIDTH) // Settings.HEIGHT
            coordY_down = x1 // Settings.WIDTH
            # calc index
            self.index1 = int((coordX_up * Settings.WORLD['width']) + coordY_up)
            self.index2 = int((coordX_down * Settings.WORLD['width']) + coordY_down)

    def highlight(self):
        '''
        This function highlights a line in red
        '''
        penThick = QPen(Qt.red, Settings.BORDER_WIDTH)
        self.setPen(penThick)
        self.changed = True

    def unhighlight(self):
        '''
        This function turns a red line back to being a whiteline
        '''
        penThick = QPen(Qt.gray, Settings.BORDER_WIDTH)
        self.setPen(penThick)
        self.changed = False

    def checkInvalidState(self, index):
        '''
        This functions checks if a state is no longer reachable by the virtual agent, because all possible transistion have turned invalid
        This only works for deterministic gridworlds
        | **Args**
        | index: Index of the state which is checked
        '''
        #Depending on the position of a state it can become unreachable after a certain number of transistions have turned invalid- e.g 2 each corner state
        counter = 0
        # For each possible action it is checked, whether the next state is still reachable and if not the counter increases
        for action in range(4):
            h = int(index / Settings.WORLD['width'])
            w = index - h * Settings.WORLD['width']
            # left
            if action == 0:
                w = max(0, w - 1)
            # up
            elif action == 1:
                h = max(0, h - 1)
            # right
            elif action == 2:
                w = min(Settings.WORLD['width'] - 1, w + 1)
            # down
            else:
                h = min(Settings.WORLD['height'] - 1, h + 1)
            nextState = int(h * Settings.WORLD['width'] + w)
            if (index, nextState) in Settings.InvalidTransistions:
                counter += 1

        # If a state reaches a specific counter number it can no longer be reached and gets added to the invalidStates

        if (index == 0 or index == Settings.WORLD["width"] - 1 or index == Settings.WORLD["states"] - 1 or index ==
                Settings.WORLD["states"] - Settings.WORLD["width"]):
            if (counter >= 2):
                Settings.InvalidStates.append(index)
                print("Invalid", index)
        elif (0 == index % Settings.WORLD["width"] or 4 == index % Settings.WORLD["width"]):
            if (counter >= 3):
                Settings.InvalidStates.append(index)
                print("Invalid", index)
        elif (index > 0 and index < Settings.WORLD["width"]) or (
                index > Settings.WORLD["states"] - Settings.WORLD["width"] and index < index == Settings.WORLD[
            "states"] - 1):
            if (counter >= 3):
                Settings.InvalidStates.append(index)
                print("Invalid", index)
        else:
            if (counter >= 4):
                Settings.InvalidStates.append(index)
                print("Invalid", index)

    def mousePressEvent(self, event):
        '''
        This function checks if a border  has been double clicked. Un/highlights the border  and
        edits the WORLD dict accordingly

        | **Args**
        | event: The mouse press event
        '''
        Settings.changed = True
        if (self.changed == False):
            self.highlight()
            #Adds the now invalid transision double to the InvalidTransistions-array, which is later fed to the update function
            Settings.InvalidTransistions.append((self.index1, self.index2))
            Settings.InvalidTransistions.append((self.index2, self.index1))
            #Checks if a state has become invalid
            self.checkInvalidState(self.index1)
            self.checkInvalidState(self.index2)
            #Updates the dictionary
            #gridTools.updateTransitions(Settings.WORLD, Settings.InvalidStates, Settings.InvalidTransistions)
            updateTransitions(Settings.WORLD, Settings.InvalidStates, Settings.InvalidTransistions)
        else:
            self.unhighlight()
            #Gets the position of the (index1, index2) double to remove it from array
            self.indexList1 = Settings.InvalidTransistions.index((self.index1, self.index2))
            Settings.InvalidTransistions.pop(self.indexList1)
            #Gets the position of the (index2, index1) double to remove it from array
            self.indexList2 = Settings.InvalidTransistions.index((self.index2, self.index1))
            Settings.InvalidTransistions.pop(self.indexList2)
            #If a state is also invald reomves it from the InvalidStates array
            if (self.index1 in Settings.InvalidStates):
                indexRemove = Settings.InvalidStates.index(self.index1)
                Settings.InvalidStates.pop(indexRemove)
            if (self.index2 in Settings.InvalidStates):
                indexRemove = Settings.InvalidStates.index(self.index2)
                Settings.InvalidStates.pop(indexRemove)

            #gridTools.updateTransitions(Settings.WORLD, Settings.InvalidStates, Settings.InvalidTransistions)
            updateTransitions(Settings.WORLD, Settings.InvalidStates, Settings.InvalidTransistions)


class Grid(QGraphicsScene):
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

        # self.terminalsSymbols = []
        # self.startingsSymbols = []
        # self.terminalsSymbols = np.zeros(Settings.WORLD['states'])
        # self.startingsSymbols = np.zeros(Settings.WORLD['states'])
        self.terminalsSymbols = np.empty((Settings.WORLD["states"], 1), QGraphicsScene)
        self.startingsSymbols = np.empty((Settings.WORLD["states"], 1), QGraphicsScene)
        # call once at creation
        # self.highlight_terminal_starting(self.terminalsSymbols, self.startingsSymbols)

    def draw_grid(self):
        '''
        This function draws the grid.
        '''
        pen = QPen(Qt.gray, Settings.BORDER_WIDTH)
        pen2 = QPen(Qt.black, Settings.BORDER_WIDTH)

        # Lines parallel to GUI x-axis(vertical)
        for x in range(1, Settings.WORLD['width']):
            xc = x * Settings.WIDTH
            for y in range(0, Settings.WORLD['height']):
                yc = y * Settings.HEIGHT

                d = yc + Settings.HEIGHT

                line = Line(xc, yc, xc, d, pen)
                if (line.index1, line.index2) in Settings.WORLD['invalidTransitions']:
                    line.highlight()

                self.lines.append(self.addItem(line))
        # Lines parallel to GUI y-axis(horizontal)
        for y in range(1, Settings.WORLD['height']):
            yc = y * Settings.HEIGHT
            for x in range(0, Settings.WORLD['width']):
                xc = x * Settings.WIDTH
                c = xc + Settings.WIDTH
                line = Line(xc, yc, c, yc, pen)
                if (line.index1, line.index2) in Settings.WORLD['invalidTransitions']:
                    line.highlight()

                self.lines.append(self.addItem(line))

        # Outer border
        self.lines.append(self.addLine(0, 0, self.width, 0, pen2))
        self.lines.append(self.addLine(0, 0, 0, self.height, pen2))
        self.lines.append(self.addLine(0, self.height, self.width, self.height, pen2))
        self.lines.append(self.addLine(self.width, 0, self.width, self.height, pen2))

    def set_visible(self, visible=True):
        '''
        Sets the grid visible/invisible

        | **Args**
        | visible: Boolean, whether the grid s visible or not
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
        This function sets the opacity for the lines

        | **Args**
        | opacity: The opacity the lines will have
        '''
        for line in self.lines:
            line.setOpacity(opacity)

    def highlight_state(self, x, y):
        '''
        This function highlights a state with a blue square

        | **Args**
        | x: State's x-coordinate
        | y: State's y-coordinate

        '''
        xc = x * Settings.HEIGHT + Settings.BORDER_WIDTH / 2
        yc = y * Settings.WIDTH + Settings.BORDER_WIDTH / 2

        brd_Wdth = Settings.BORDER_WIDTH

        # removes pieces old highlight
        self.removeItem(self.highlight)
        self.highlight = self.addRect(yc, xc,
                                      Settings.WIDTH - brd_Wdth,
                                      Settings.HEIGHT - brd_Wdth,
                                      self.pen, self.brush)
        # removes artefacts of old highlights
        self.update()

    def highlight_terminal_starting(self, terminals, startings):
        '''
        This function highlights the states according to theterminal and
        starting state list in the  WORLD directory

        | **Args**
        | terminals: The terminal state list in the WORLD dict
        | startings: The starting state  list in the WOLRD dict
        '''
        #print("test highlight terminal starting")
        # delete all symbols created before and then empty the array
        for index in range(len(self.terminalsSymbols)):
            self.removeItem(self.terminalsSymbols[int(index), 0])
        # self.terminalsSymbols = np.array([])

        for index in range(len(self.startingsSymbols)):
            self.removeItem(self.startingsSymbols[int(index), 0])
        # self.startingsSymbols = np.array([])
        # create symbol at position x,y for all terminals and startings
        for index in range(len(terminals)):
            if terminals[index] == 1:
                # calc x,y position
                y = (index // Settings.WORLD["width"]) * Settings.WIDTH
                x = (index % Settings.WORLD["width"]) * Settings.HEIGHT
                # place X at position (x,y)
                self.terminalsSymbols[int(index), 0] = self.addText("X")
                self.terminalsSymbols[int(index), 0].setPos(x, y)
        for index in startings:
            # print("q" + str(index))
            # calc x,5 position
            y = (index // Settings.WORLD["width"]) * Settings.WIDTH
            x = (index % Settings.WORLD["width"]) * Settings.HEIGHT
            # print(str(x) + " xx " + str(y))
            # place S at position (x,y)
            self.startingsSymbols[int(index), 0] = self.addText("S")
            self.startingsSymbols[int(index), 0].setPos(x, y)
            # print(str(self.startingsSymbols))
        self.update()

    def mousePressEvent(self, event):
        '''
        This function determines which state or if a border has been clicked on
        
        | **Args**
        | event:                       The triggering event.
        '''
        
        # Gets mouse position within scene
        posX = event.scenePos().x()
        posY = event.scenePos().y()

        # check if mouse within scene at all
        if posX >= 0 and posY >= 0 and posX <= self.width and posY <= self.height:
            absText = "a: " + str(posX) + " " + str(posY)

            # check if mouse position is on border (rel = relative pos in state)
            relX = (posX + Settings.BORDER_WIDTH / 2) % Settings.WIDTH
            relY = (posY + Settings.BORDER_WIDTH / 2) % Settings.HEIGHT

            if relX <= Settings.BORDER_WIDTH + 0.5 or relY <= Settings.BORDER_WIDTH + 0.5:
                relText = "r: " + str(relX) + " " + str(relY)
                text = "Border: " + absText + " -> " + relText
                #print(text)
            else:
                # translate to coordinate
                coordX = int(posY // Settings.HEIGHT)
                coordY = int(posX // Settings.WIDTH)
                # calc index
                index = (coordX * Settings.WORLD['width']) + coordY
                # Console output of state coordinates
                coordText = "(" + str(coordX) + "," + str(coordY) + ")"
                text = "State: " + absText + " -> c: " + coordText
                #print(text)

                # highlight current state
                self.highlight_state(coordX, coordY)

                # update state info
                self.parent.changeStateInfo(coordText, index)


class GridViewer(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setScene(self.scene)
        self.parent = parent
        self.setSceneRect(self.sceneRect())
        
        # Zoom counter and maximum zoom count
        self.zoom = 0
        self.maxZoom = 0

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        

    def wheelEvent(self, event):
        '''
        This function scales the view depending on the wheel event
        
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

        if self.zoom > 0 and self.zoom <= self.maxZoom:
            self.scale(factor, factor)
        elif self.zoom == 0:
            # fits scene in view on maximum zoom out
            self.fitInView(self.scene.itemsBoundingRect(),Qt.KeepAspectRatio)
        elif self.zoom >= self.maxZoom:
            # limits zooming in
            self.zoom = self.maxZoom
        else:
            self.zoom = 0
        
        self.scene.update()
            
    def showEvent(self, event):
        '''
        This function fits the scene in view as soon as the view is shown
        
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
            self.maxZoom = math.floor(math.log(maxFactor,5/4))

class StateInformation(QWidget):
    def __init__(self, parent=None):
        super(StateInformation, self).__init__(parent=parent)
        # WIDGETS
        # State Information
        heading = QLabel()
        heading.setText("STATE INFORMATION")
        # Index
        self.index = QLabel()
        self.indexValue = 0
        # Coordinates
        self.coordinates = QLabel()
        # Reward
        validator = QIntValidator(0, 2147483647)
        reward = QLabel()
        reward.setText("Reward:")
        self.rewardValue = QLineEdit()
        self.rewardValue.setText("0")
        self.rewardValue.setValidator(validator)
        # self.actualValue = 0
        rewardLine = QHBoxLayout()
        rewardLine.addWidget(reward)
        rewardLine.addWidget(self.rewardValue)
        # Radio Button, Starting, Terminal, Goal
        radioLabel = QLabel("Set state type:")
        self.terminalBtn = QRadioButton("Terminal")
        self.startBtn = QRadioButton("Starting")
        self.noneBtn = QRadioButton("None")
        # Apply Button
        self.applyBtn = QPushButton()
        self.applyBtn.setText("Apply Changes")
        self.applyBtn.clicked.connect(self.updateState)
        # Extended Settings
        self.extendedBtn = QPushButton()
        self.extendedBtn.setText("Advanced Settings")
        self.extendedBtn.clicked.connect(self.openAdvanced)

        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(heading)
        layout.addWidget(self.index)
        layout.addWidget(self.coordinates)
        self.layout().addLayout(rewardLine)
        layout.addWidget(radioLabel)
        layout.addWidget(self.terminalBtn)
        layout.addWidget(self.startBtn)
        layout.addWidget(self.noneBtn)

        layout.addWidget((self.extendedBtn))
        layout.addWidget(self.applyBtn)

    def changeStateInfo(self, textCoord, index):
        '''
        this function is used  to  update the state information panel
        text according to the selected state.

        | **Args**
        | textCoord: The  states coordinates
        | index: The states index
        '''
        self.index.setText("Index: " + str(int(index)))
        self.indexValue = index
        self.coordinates.setText("Coordinates: " + textCoord)
        self.rewardValue.setText(str(int(Settings.WORLD["rewards"][int(index)])))

        if (Settings.WORLD["terminals"][int(index)] == 1):
            self.terminalBtn.click()
            print("terminal state")
        elif index in Settings.WORLD["startingStates"]:
            self.startBtn.click()
        else:
            self.noneBtn.click()

    def updateState(self):
        '''
        This function is used to update the selected states properties inside
        the WORLD directory.
        '''
        Settings.changed = True

        rValue = int(self.rewardValue.text())
        # checks that the  reward value is a positive number
        if (rValue >= 0 and rValue <= 2147483647):
            #gridworld_tools.updateState(Settings.WORLD, int(self.indexValue), rValue, self.terminalBtn.isChecked(),
            #                            self.startBtn.isChecked(), self.noneBtn.isChecked())
            updateState(Settings.WORLD, int(self.indexValue), rValue, self.terminalBtn.isChecked(),
                                        self.startBtn.isChecked(), self.noneBtn.isChecked())
            # updates state visualization
            self.parent().parent().highlight_terminal_starting(Settings.WORLD["terminals"],
                                                               Settings.WORLD["startingStates"])
        else:
            print("Please enter a positive number for reward")

    def openAdvanced(self):
        '''
        This function opens the advanced settings menu
        '''
        Settings.index = int(self.indexValue)
        self.advWindow = AdvancedSettingsWindow()
        self.advWindow.show()

class AdvancedSettingsWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QMainWindow.__init__(self)
        self.setWindowTitle('Advanced Settings')
        self.resize(500, 500)

        self.advancedWidget = AdvancedSettingsWidget()
        self.setCentralWidget(self.advancedWidget)

class AdvancedSettingsWidget(QWidget):
    '''
    The Widget, that handles the table for the transition probabilities
    '''
    def __init__(self):
        QWidget.__init__(self)
        #The table with 4 columns and as many roows as there are states
        self.tableWidget = QTableWidget(Settings.WORLD["states"], 4, self)
        self.tableWidget.setHorizontalHeaderLabels([str(i) for i in range(self.tableWidget.columnCount())])
        self.tableWidget.setVerticalHeaderLabels([str(i) for i in range(self.tableWidget.rowCount())])
        self.rows = self.tableWidget.rowCount()
        self.columns = self.tableWidget.columnCount()
        #This fills the table with QLineEdits, which are filled with the data from the ["sas"] array
        for column in range(self.columns):
            for row in range(self.rows):
                print("1")
                item = float(Settings.WORLD["sas"][int(Settings.index)][column][row])

                cellValue = QLineEdit()
                cellValue.setText(str(item))
                print("2")
                validator = QDoubleValidator(0.0, 1.0, 2)

                cellValue.setFrame(False)
                cellValue.setValidator(validator)
                self.tableWidget.setCellWidget(row, column, cellValue)

        #Save Button
        saveBtn = QPushButton()
        saveBtn.setText("Save Changes")
        saveBtn.clicked.connect(self.saveChanges)
        #Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.tableWidget)
        self.layout.addWidget(saveBtn)

    def saveChanges(self):
        '''
        This function closes the advanced settings window and saves the changes made
        '''
        #Array to save the changed transition probabilities and check if the entered data is valid - smaller than 1
        probabilityChanges = np.zeros((4, Settings.WORLD["states"]))
        for column in range(self.columns):
            for row in range(self.rows):
                probabilityChanges[column][row] = float(self.tableWidget.cellWidget(row, column).text())

        savable = False
        # Checks if the entered data for each columns does not exceed 1
        count = 0
        for column in range(self.columns):
            if (count > 1):
                print("Your probabilities exceed 100%")
                savable = False
                break
            else:
                savable = True
            count = 0
            for row in range(self.rows):
                count += probabilityChanges[column][row]

        if (savable == True):
            #gridworld_tools.updateProbability(Settings.WORLD, Settings.index, probabilityChanges)
            updateProbability(Settings.WORLD, Settings.index, probabilityChanges)
            self.parent().close()


class CentralPanel(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # Grid and View
        self.scene1 = Grid(self)
        self.view = GridViewer(self.scene1, self)

        self.scene1.highlight_terminal_starting(Settings.WORLD["terminals"], Settings.WORLD["startingStates"])
        # Sidebar
        # self.lblState = QLabel('State-Info:')
        self.stateMenu = StateInformation(self)

        # splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(self.stateMenu)  # sidebar here
        splitter.setStretchFactor(0, 10)

        # add splitter(Grid & InfoBar) to layout
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(splitter)
        self.setLayout(mainLayout)

    def changeStateInfo(self, textCoord, index):
        '''
        Used to pass the fucntion to the children of Central Panel.

        | **Args**
        | textCoord:  states coordinates
        | index: states index
        '''
        self.stateMenu.changeStateInfo(textCoord, index)

    def highlight_terminal_starting(self, terminals, startings):
        '''
        Used to pass the function to the state info menu

        | **Args**
        | terminals: worlds terminal states
        | startings: worlds starting states
        '''
        self.scene1.highlight_terminal_starting(terminals, startings)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QMainWindow.__init__(self)
        self.setWindowTitle('Gridworld Editor')

        self.create_MenuBar()

        self.CenterPanel = CentralPanel()
        self.setCentralWidget(self.CenterPanel)

        # Window size
        self.resize(1000, 563)

    def create_MenuBar(self):
        '''
        This function creates the menu bar.
        '''
        # Define actions
        actionNew = QAction('&New', self)
        actionNew.setShortcut("Ctrl+N")
        actionNew.triggered.connect(self.new)

        actionOpen = QAction('&Open', self)
        actionOpen.setShortcut("Ctrl+O")
        actionOpen.triggered.connect(self.load)

        actionSave = QAction('&Save', self)
        actionSave.setShortcut("Ctrl+S")
        actionSave.triggered.connect(self.saveCurrentFile)

        actionSaveAs = QAction('&Save as', self)
        actionSaveAs.setShortcut("Ctrl+Shift+S")
        actionSaveAs.triggered.connect(self.saveAs)

        actionQuit = QAction('&Quit', self)
        actionQuit.setShortcut("Ctrl+Q")
        actionQuit.triggered.connect(self.close)

        actionInfo = QAction('&Info', self)
        actionInfo.setShortcut("Ctrl+I")
        actionInfo.triggered.connect(self.info)

        # Create menuBar
        self.MenuBar = self.menuBar()
        # Add menus to menuBar
        fileMenu = self.MenuBar.addMenu('&File')

        infoMenu = self.MenuBar.addMenu('&Info')
        # Add actions to menu
        fileMenu.addAction(actionNew)
        fileMenu.addAction(actionOpen)
        fileMenu.addAction(actionSave)
        fileMenu.addAction(actionSaveAs)
        fileMenu.addAction(actionQuit)

        infoMenu.addAction(actionInfo)

    def getOpenFile(self):
        '''
        This function calls the Windows file explorer to get an existing file path.
        '''
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "Pickle files (*.pkl)")
        fname = fname[0]

        print(fname)
        return fname

    def getSaveFile(self):
        '''
        This function calls the Windows file explorer to get any file path.
        '''
        fname = QFileDialog.getSaveFileName(self, 'Open file',
                                            'c:\\', "Pickle files (*.pkl)")
        fname = fname[0]
        # To-Do check .pkl ending

        print(fname)
        return fname

    def new(self):
        '''
        This function returns the user to the starting window.
        '''
        print("NEW")
        if not self.unsavedChanges():
            Settings.changed = False
            Settings.file = ""
            Settings.InvalidTransistions = []
            Settings.InvalidStates = []

            self.b = StartWindow()
            self.b.show()
            self.close()

    def load(self):
        '''
        This function asks the user to select a file and opens it.
        '''
        print("LOAD")
        if not self.unsavedChanges():
            fname = self.getOpenFile()
            if fname != "":

                Settings.changed = False
                Settings.file = fname
                Settings.WORLD = pickle.load(open(fname, 'rb'))
                Settings.InvalidTransistions = Settings.WORLD["invalidTransitions"]
                Settings.InvalidStates = Settings.WORLD["invalidStates"]
                self.b = MainWindow()
                self.b.show()
                self.close()
                print("LOAD SUCCESS")
            else:
                print("LOAD FAIL")

    def save(self, fname):
        '''
        This function saves the current gridworld in a file

        | **Args**
        | fname:                       File name/Target file.
        '''
        print("SAVE")
        if fname != "":
            pickle.dump(Settings.WORLD, open(fname, 'wb'))
            Settings.changed = False
            print("SAVE SUCCESS")
        else:
            print("SAVE FAIL")

    def saveAs(self):
        '''
        This function asks the user to select a file save the gridworld in it.
        '''
        print("SAVE AS")
        fname = self.getSaveFile()
        if fname != "":
            self.save(fname)

    def saveCurrentFile(self):
        '''
        This function saves the gridworld in its currently selected file.
        If no file is currently selected the user is asked to choose one.
        '''
        print("SAVE CURRENT FILE")
        if Settings.file == "":
            Settings.file = self.getSaveFile()
        self.save(Settings.file)

    def unsavedChanges(self):
        '''
        This function opens a dialog window to asks the user to save the
        gridworld if there are unsaved changes.
        '''
        # Returns False if there are no unsaved changes/if these should be ignored
        if Settings.changed == True:
            print("UNSAVED CHANGES")
            self.b = unsavedChangesDialog(self)

            if self.b.exec():
                return False
            else:
                return True

        return False
    
    def info(self):
        '''
        This function opens the info dialog
        '''
        self.b = infoDialog(self)
        self.b.exec()

    def closeEvent(self, event):
        '''
        This function closes the programm.
        '''
        if not self.unsavedChanges():
            super().closeEvent(event)
        else:
            event.ignore()

class unsavedChangesDialog(QDialog):
    def __init__(self, parent=None):
        super(unsavedChangesDialog, self).__init__(parent)
        QDialog.__init__(self)
        self.parent = parent

        self.layout = QGridLayout()
        self.setWindowTitle('Unsaved Changes')

        self.label = QLabel()
        self.label.setText("The file has been modified. Do you want to save the changes?")
        self.b1 = QPushButton("Yes")
        self.b2 = QPushButton("No")
        self.b3 = QPushButton("Cancel")

        self.layout.addWidget(self.label, 0, 0, 1, 3)
        self.layout.addWidget(self.b1, 1, 0, 1, 1)
        self.layout.addWidget(self.b2, 1, 1, 1, 1)
        self.layout.addWidget(self.b3, 1, 2, 1, 1)
        self.b1.clicked.connect(self.saveCurrentFile)
        self.b2.clicked.connect(self.accept)
        self.b3.clicked.connect(self.reject)
        self.setLayout(self.layout)

    def saveCurrentFile(self):
        '''
        This function saves the gridworld in its currently selected file.
        If no file is currently selected the user is asked to choose one.
        '''
        self.hide()
        self.parent.saveCurrentFile()
        if Settings.changed == False:
            self.accept()
        else:
            self.reject()
            
class infoDialog(QDialog):
    def __init__(self, parent=None):
        super(infoDialog, self).__init__(parent)
        QDialog.__init__(self)
        self.parent = parent

        self.layout = QGridLayout()
        self.setWindowTitle('Info')

        self.label = QLabel("Developed by : William Forchap, Kilian Kandt, Marius Tenhumberg")
        self.label2 = QLabel("Supervised by : Nicolas Diekmann")
        self.b1 = QPushButton("Nice")

        self.layout.addWidget(self.label, 0, 0, 1, 1)
        self.layout.addWidget(self.label2, 1, 0, 1, 1)
        self.layout.addWidget(self.b1, 2, 0, 1, 1)
        self.b1.clicked.connect(self.accept)
        self.setLayout(self.layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    a = StartWindow()
    a.show()

    app.exec()
