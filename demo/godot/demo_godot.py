"""
This demo simulation showcases how to instatiate and control
the Godot simulator via the GodotSimulator class.
A scene consisting of a square room is loaded and the virtual
agent (represented as a white cube) moves in a circular motion
through the room while the color of a light source is constantly
being changed.
Additionally, information about the rendered image and
simulation objects is printed.
"""

# basic imports
import sys
import time
import numpy as np
# CoBeL-RL
from cobel.interface import GodotSimulator


def single_run() -> None:
    """
    This function represents one demo run.
    """
    # init simulator
    executable: None | str = None
    if '--executable' in sys.argv:
        executable = sys.argv[sys.argv.index('--executable') + 1]
    simulator = GodotSimulator('room.tscn', executable=executable)
    # print image info
    print('\nImage Info:', simulator.image_info, '\n')
    # test echo command
    for i in range(10):
        simulator.echo(str(i))
    # print object info
    print('\nSimulation objects:')
    for object_id, obj in simulator.objects.items():
        print([object_id, obj['name'], obj['type']])
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
    colors += [[255, 255 - i, 255 - i] for i in range(256)]
    colors += [[255 - i, i, 0] for i in range(256)]
    colors += [[0, 255 - i, i] for i in range(256)]
    # drive the virtual agent and environment
    for i in range(len(colors)):
        time.sleep(0.01)
        # change the spotlight's color
        simulator.set_illumination('SpotLight3D', np.array(colors[i]))
        # move the virtual agent
        simulator.move_agent(positions[i, 0], positions[i, 1], np.rad2deg(rotations[i]))
    # stop simulation
    simulator.stop()


if __name__ == '__main__':
    single_run()
