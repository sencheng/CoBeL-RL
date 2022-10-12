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
    modules['world'] = FrontendGodotInterface('room')
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    
    rotations = np.linspace(0, 2 * np.pi, 255)
    for i in range(255):
        modules['world'].set_illumination('SpotLight', np.array([255, 255-i, 255-i]))
        time.sleep(0.05)
        modules['world'].step_simulation_without_physics(0, 0, rotations[i])
        time.sleep(0.05)
        modules['observation'].update()
        if hasattr(qt.QtGui, 'QApplication'):
            qt.QtGui.QApplication.instance().processEvents()
        else:
            qt.QtWidgets.QApplication.instance().processEvents()
    for i in range(255):
        modules['world'].set_illumination('SpotLight', np.array([255-i, i, 0]))
        time.sleep(0.05)
        modules['world'].step_simulation_without_physics(0, 0, rotations[i])
        time.sleep(0.05)
        modules['observation'].update()
        if hasattr(qt.QtGui, 'QApplication'):
            qt.QtGui.QApplication.instance().processEvents()
        else:
            qt.QtWidgets.QApplication.instance().processEvents()
    for i in range(255):
        modules['world'].set_illumination('SpotLight', np.array([0, 255-i, i]))
        time.sleep(0.05)
        modules['world'].step_simulation_without_physics(0, 0, rotations[i])
        time.sleep(0.05)
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
