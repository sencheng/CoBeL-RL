# basic imports
import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore
# typing
from numpy.typing import NDArray

WallList = list[tuple[NDArray, bool]]
PillarList = list[NDArray]


def state_coordinates(
    state: int, width: int, height: int, state_size: float
) -> NDArray:
    """
    This function computes the coordinates of a gridworld state.

    Parameters
    ----------
    state : int
        The gridworld state index.
    width : int
        The width of the gridworld environment.
    height : int
        The height of the gridworld environment.
    state_size : float
        The size (or scale) of the state.

    Returns
    -------
    coordinates : NDArray
        The 3-dimensional coordinates of the gridworld state (z-coordinates is 0).
    """
    y = height - int(state / width) - 0.5
    x = state % width + 0.5

    return np.array([x, y, 0.0]) * state_size


def retrieve_wall_info(
    invalid_transitions: list[tuple[int, int]],
    width: int,
    height: int,
    state_size: float,
    include_perimeter: bool = True,
) -> tuple[WallList, PillarList]:
    """
    This function retrieves the necessary wall information
    from a list of invalid transitions.

    Parameters
    ----------
    invalid_transitions : list of 2-tuple of int
        A list invalid transition tuples.
    width : int
        The width of the gridworld environment.
    height : int
        The height of the gridworld environment.
    state_size : float
        The size (or scale) of the state.
    include_perimeter : bool, default=True
        If True, perimeter walls will be included.

    Returns
    -------
    walls : WallList
        A list containing the position and rotation information for each wall.
    pillars : PillarList
        A list containing the position information for each pillar.
    """
    walls: WallList = []
    pillars: PillarList = []
    for transition in list(
        set([tuple(set(transition)) for transition in invalid_transitions])
    ):
        # wall info
        rotation = not np.abs(transition[0] - transition[1]) > 1
        position = (
            state_coordinates(transition[0], width, height, state_size)
            + state_coordinates(transition[1], width, height, state_size)
        ) / 2
        walls.append((position, rotation))
        # pillar info
        pillar_offset = np.array([not rotation, rotation, 0]) * state_size / 2
        if len(pillars) == 0:
            pillars.append(position - pillar_offset)
            pillars.append(position + pillar_offset)
        else:
            if (
                np.amax(np.sum(np.array(pillars) == position - pillar_offset, axis=1))
                < 3
            ):
                pillars.append(position - pillar_offset)
            if (
                np.amax(np.sum(np.array(pillars) == position + pillar_offset, axis=1))
                < 3
            ):
                pillars.append(position + pillar_offset)
    # perimeter walls and pillars
    if include_perimeter:
        pillars += [
            np.array([0, 0, 0]),
            np.array([width, 0, 0]) * state_size,
            np.array([0, height, 0]) * state_size,
            np.array([width, height, 0]) * state_size,
        ]
        for i in range(width):
            walls.append((np.array([i + 0.5, 0, 0]) * state_size, False))
            walls.append((np.array([i + 0.5, height, 0]) * state_size, False))
            top = np.array([i, 0, 0]) * state_size
            bottom = np.array([i, height, 0]) * state_size
            if np.amax(np.sum(np.array(pillars) == top, axis=1)) < 3:
                pillars.append(top)
            if np.amax(np.sum(np.array(pillars) == bottom, axis=1)) < 3:
                pillars.append(bottom)
        for i in range(height):
            walls.append((np.array([0, i + 0.5, 0]) * state_size, True))
            walls.append((np.array([width, i + 0.5, 0]) * state_size, True))
            left = np.array([0, i, 0]) * state_size
            right = np.array([width, i, 0]) * state_size
            if np.amax(np.sum(np.array(pillars) == left, axis=1)) < 3:
                pillars.append(left)
            if np.amax(np.sum(np.array(pillars) == right, axis=1)) < 3:
                pillars.append(right)

    return walls, pillars


def generate_walls(
    walls: WallList,
    pillars: PillarList,
    width: int,
    height: int,
    state_size: float,
    wall_height: float,
    wall_depth: float,
) -> list[NDArray]:
    """
    This function generates a list of wall objects.

    Parameters
    ----------
    walls : WallList
        A list containing the position and rotation information
        of the wall objects that will be generated.
    pillars : PillarList
        A list containing the position information of the
        pillar objects that will be generated.
    width : int
        The width of the gridworld environment.
    height : int
        The height of the gridworld environment.
    state_size : float
        The size (or scale) of the state.
    wall_height : float
        The height of the generated wall objects.
    wall_depth : float
        The depth of the generated wall objects.
        Has to be smaller than half of the state size.

    Returns
    -------
    meshes : list of NDArray
        A list of vertices that define walls and pillars.
    """
    # check validity of parameters
    assert wall_depth < state_size / 2 and wall_height > 0.0
    # compute remaining parameters
    wall_width = state_size - wall_depth
    # prepare wall and pillar templates
    cube = np.array(
        [
            [x, y, z]
            for x, y, z in itertools.product([-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0])
        ]
    )
    wall_template = cube * np.array([wall_width / 2, wall_depth / 2, wall_height])
    pillar_template = cube * np.array([wall_depth / 2, wall_depth / 2, wall_height])
    # compute wall vertices
    meshes: list[NDArray] = []
    for wall in walls:
        meshes.append(np.copy(wall_template))
        # rotate if the wall is vertical
        if wall[1]:
            r = R.from_euler('z', 90, True)
            meshes[-1] = r.apply(meshes[-1]).round(15)
        # translate wall
        meshes[-1] += wall[0]
    # compute pillars vertices
    for pillar in pillars:
        meshes.append(pillar_template + pillar)

    return meshes


def export_as_obj(
    meshes: list[NDArray], width: int, height: int, state_size: float, file_name: str
) -> None:
    """
    This function exports the gridworld as 3D object in the Wavefront (obj) format.
    Vertices are written in the format (z+ up, y+ front).

    Parameters
    ----------
    walls : list of NDArray
        A list of wall objects.
    width : int
        The width of the gridworld environment.
    height : int
        The height of the gridworld environment.
    state_size : float
        The size (or scale) of the state.
    file_name : str
        The name that the file will be saved as.
    """
    with open(file_name, 'w+') as file:
        vertex_offset = 1
        object_name = 'o border_object_%0' + '%dd\n' % (len(str(len(meshes))) + 1)
        # write wall vertices
        for m, mesh in enumerate(meshes):
            # write object name
            file.write(object_name % m)
            for face in [
                [0, 1, 5, 4],
                [0, 1, 3, 2],
                [4, 5, 7, 6],
                [2, 3, 7, 6],
                [1, 5, 7, 3],
                [0, 4, 6, 2],
            ]:
                for vertex in face:
                    file.write('v %f %f %f\n' % tuple(mesh[vertex]))
                file.write('f %d %d %d %d\n' % tuple(np.arange(4) + vertex_offset))
                vertex_offset += 4
        # write floor vertices
        file.write('o floor\n')
        floor = np.array(
            [
                [x, y, z]
                for x, y, z in itertools.product(
                    [0.0, state_size], [0.0, state_size], [0.0]
                )
            ]
        ) * np.array([width, height, 0])
        for vertex in [0, 1, 3, 2]:
            file.write('v %f %f %f\n' % tuple(floor[vertex]))
        file.write('f %d %d %d %d\n' % tuple(np.arange(4) + vertex_offset))
        vertex_offset += 4
