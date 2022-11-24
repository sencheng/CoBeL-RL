# basic imports
import time
import numpy as np
import pyqtgraph as qg
import PyQt5 as qt
# framework imports
from cobel.frontends.frontends_godot import FrontendGodotInterface
from cobel.observations.image_observations import ImageObservationBaseline

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: Godot Interface')
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendGodotInterface('room.tscn')
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['observation'].format = 'rgb'
    
    # test echo command
    for i in range(10):
        modules['world'].echo(str(i))
    
    # print object info
    print('\nSimulation objects:')
    for object_ID in modules['world'].objects:
        print([object_ID, modules['world'].objects[object_ID]['name'], modules['world'].objects[object_ID]['type']])
    
    # define pose information
    rotations = np.arange(256) * 2 * np.pi / 256
    positions = np.zeros((256, 2))
    positions[:, 0] = np.sin(rotations)
    positions[:, 1] = np.cos(rotations)
    positions *= 0.5
    rotations = np.tile(rotations, 3)
    positions = np.tile(positions, (3, 1))
    # define color information
    colors = []
    colors += [[255, 255-i, 255-i] for i in range(256)]
    colors += [[255-i, i, 0] for i in range(256)]
    colors += [[0, 255-i, i] for i in range(256)]
    
    # drive the virtual agent and environment 
    for i in range(len(colors)):
        time.sleep(0.01)
        # change the spotlight's color
        modules['world'].set_illumination('SpotLight', np.array(colors[i]))
        # move the virtual agent
        modules['world'].step_simulation_without_physics(positions[i, 0], positions[i, 1], np.rad2deg(rotations[i]))
        # update observation and visualization 
        modules['observation'].update()
        if hasattr(qt.QtGui, 'QApplication'):
            qt.QtGui.QApplication.instance().processEvents()
        else:
            qt.QtWidgets.QApplication.instance().processEvents()
    
    # stop simulation
    modules['world'].stop_godot()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    single_run()
