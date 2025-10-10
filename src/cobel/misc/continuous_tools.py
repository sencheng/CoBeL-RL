# basic imports
import numpy as np
import shapely as sh  # type: ignore
from shapely.affinity import rotate, translate  # type: ignore
# typing
from typing import Literal
from numpy.typing import NDArray

ContinuousTemplate = tuple[sh.Polygon, sh.Polygon, list[sh.Polygon], NDArray]


def make_t_maze(
    stem_length: float,
    arm_length: float,
    corridor_width: float,
    goal_arm: Literal['left', 'right', 'none'] = 'right',
    reward: float = 1,
) -> ContinuousTemplate:
    """
    This function builds a T-maze environment.

    Parameters
    ----------
    stem_length : float
        The T-maze's stem length.
    arm_length : float
        The T-maze's arm length.
    corridor_width : float
        The T-maze's corridor width.
    goal_arm : str, default='right'
        The rewarded arm (left, right, none).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    room : sh.Polygon
        A polygon representing the environmental borders.
    spawn : sh.Polygon
        A polygon representing the agent's starting area.
    obstacles : list of sh.Polygon
        A list of polygons representing the environment's obstacles.
    rewards : NDArray
        A numpy array encoding the reward positions and magnitudes.
    """
    assert corridor_width > 0, 'Corridor width must be positive!'
    # ensure that stem and arm are at least as long as the corridor is wide
    stem_length = max(corridor_width, stem_length)
    arm_length = max(corridor_width, arm_length)
    height, width = stem_length + corridor_width, arm_length * 2 + corridor_width
    # environmental borders
    borders = np.array(
        [
            [0, height],
            [width, height],
            [width, height - corridor_width],
            [width - arm_length, height - corridor_width],
            [width - arm_length, 0],
            [arm_length, 0],
            [arm_length, height - corridor_width],
            [0, height - corridor_width],
            [0, height],
        ]
    )
    # reward location
    rewards = np.array([])
    if goal_arm in ['left', 'right']:
        offset = (goal_arm == 'right') * arm_length * 2
        rewards = np.array(
            [[corridor_width / 2 + offset, height - corridor_width / 2, reward]]
        )
    # spawn area
    spawn = np.array(
        [
            [arm_length, 0],
            [arm_length + corridor_width, 0],
            [arm_length + corridor_width, corridor_width],
            [arm_length, corridor_width],
            [arm_length, 0],
        ]
    )

    return sh.Polygon(borders), sh.Polygon(spawn), [], rewards


def make_double_t_maze(
    stem_length: float,
    arm_length: float,
    corridor_width: float,
    goal_arm: Literal[
        'left-left', 'left-right', 'right-left', 'right-right', 'none'
    ] = 'right-right',
    reward: float = 1,
) -> ContinuousTemplate:
    """
    This function builds a double T-maze environment.

    Parameters
    ----------
    stem_length : float
        The T-maze's stem length.
    arm_length : float
        The T-maze's arm length.
    corridor_width : float
        The T-maze's corridor width.
    goal_arm : str, default='right-right'
        The rewarded arm (left-left, left-right, right-left, right-right, none).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    room : sh.Polygon
        A polygon representing the environmental borders.
    spawn : sh.Polygon
        A polygon representing the agent's starting area.
    obstacles : list of sh.Polygon
        A list of polygons representing the environment's obstacles.
    rewards : NDArray
        A numpy array encoding the reward positions and magnitudes.
    """
    assert corridor_width > 0, 'Corridor width must be positive!'
    # ensure that stem and arm are at least as long as the corridor is wide
    stem_length = max(corridor_width, stem_length)
    arm_length = max(corridor_width * 2, arm_length)
    height, width = 2 * stem_length + corridor_width, arm_length * 4 - corridor_width
    # environmental borders
    borders = np.array(
        [
            [0, height],
            [2 * arm_length - corridor_width, height],
            [2 * arm_length - corridor_width, height - corridor_width],
            [arm_length, height - corridor_width],
            [arm_length, height - stem_length],
            [3 * arm_length - corridor_width, height - stem_length],
            [3 * arm_length - corridor_width, height - corridor_width],
            [2 * arm_length, height - corridor_width],
            [2 * arm_length, height],
            [width, height],
            [width, height - corridor_width],
            [width - arm_length + corridor_width, height - corridor_width],
            [
                width - arm_length + corridor_width,
                height - stem_length - corridor_width,
            ],
            [2 * arm_length, height - stem_length - corridor_width],
            [2 * arm_length, 0],
            [2 * arm_length - corridor_width, 0],
            [2 * arm_length - corridor_width, height - stem_length - corridor_width],
            [arm_length - corridor_width, height - stem_length - corridor_width],
            [arm_length - corridor_width, height - corridor_width],
            [0, height - corridor_width],
            [0, height],
        ]
    )
    # reward location
    rewards = np.array([])
    if goal_arm in ['left-left', 'left-right', 'right-left', 'right-right']:
        first, second = goal_arm.split('-')
        offset = (first == 'right') * 2 * arm_length + (second == 'right') * 2 * (
            arm_length - corridor_width
        )
        rewards = np.array(
            [[corridor_width / 2 + offset, height - corridor_width / 2, reward]]
        )
    # spawn area
    spawn = np.array(
        [
            [2 * arm_length - corridor_width, 0],
            [2 * arm_length, 0],
            [2 * arm_length, corridor_width],
            [2 * arm_length - corridor_width, corridor_width],
            [2 * arm_length - corridor_width, 0],
        ]
    )

    return sh.Polygon(borders), sh.Polygon(spawn), [], rewards


def make_two_sided_t_maze(
    stem_length: float,
    arm_length: float,
    corridor_width: float,
    goal_arm: Literal[
        'left-left', 'left-right', 'right-left', 'right-right', 'none'
    ] = 'right-right',
    reward: float = 1,
) -> ContinuousTemplate:
    """
    This function builds a two-sided T-maze environment.

    Parameters
    ----------
    stem_length : float
        The T-maze's stem length.
    arm_length : float
        The T-maze's arm length.
    corridor_width : float
        The T-maze's corridor width.
    goal_arm : str, default='right-right'
        The rewarded arm (left-left, left-right, right-left, right-right, none).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    room : sh.Polygon
        A polygon representing the environmental borders.
    spawn : sh.Polygon
        A polygon representing the agent's starting area.
    obstacles : list of sh.Polygon
        A list of polygons representing the environment's obstacles.
    rewards : NDArray
        A numpy array encoding the reward positions and magnitudes.
    """
    assert corridor_width > 0, 'Corridor width must be positive!'
    # ensure that stem and arm are at least as long as the corridor is wide
    stem_length = max(corridor_width, stem_length)
    arm_length = max(corridor_width, arm_length)
    height, width = 2 * arm_length + corridor_width, 2 * corridor_width + stem_length
    # environmental borders
    borders = np.array(
        [
            [0, height],
            [corridor_width, height],
            [corridor_width, height - arm_length],
            [width - corridor_width, height - arm_length],
            [width - corridor_width, height],
            [width, height],
            [width, 0],
            [width - corridor_width, 0],
            [width - corridor_width, arm_length],
            [corridor_width, arm_length],
            [corridor_width, 0],
            [0, 0],
            [0, height],
        ]
    )
    # reward location
    rewards = np.array([])
    if goal_arm in ['left-left', 'left-right', 'right-left', 'right-right']:
        first, second = goal_arm.split('-')
        offset = np.array(
            [
                (first == 'right') * (width - corridor_width),
                (goal_arm in ['left-right', 'right-left']) * (height - corridor_width),
            ]
        )
        rewards = np.array(
            [[corridor_width / 2 + offset[0], corridor_width / 2 + offset[1], reward]]
        )
    # spawn area
    s_x, s_y = (width - corridor_width) / 2, arm_length
    spawn = np.array(
        [
            [s_x, s_y],
            [s_x + corridor_width, s_y],
            [s_x + corridor_width, s_y + corridor_width],
            [s_x, s_y + corridor_width],
            [s_x, s_y],
        ]
    )

    return sh.Polygon(borders), sh.Polygon(spawn), [], rewards


def make_eight_maze(
    center_height: float,
    lap_width: float,
    corridor_width: float,
    goal_arm: Literal['left', 'right'] = 'right',
    reward: float = 1,
) -> ContinuousTemplate:
    """
    This function builds an 8-maze environment.

    Parameters
    ----------
    center_height : float
        The 8-maze's center length.
    lap_width : float
        The 8-maze's lap width.
    corridor_width : float
        The 8-maze's corridor width.
    goal_arm : str, default='right'
        The rewarded lap (left, right, none).
    reward : float, default=1.
        The reward provided at the rewarded lap.

    Returns
    -------
    room : sh.Polygon
        A polygon representing the environmental borders.
    spawn : sh.Polygon
        A polygon representing the agent's starting area.
    obstacles : list of sh.Polygon
        A list of polygons representing the environment's obstacles.
    rewards : NDArray
        A numpy array encoding the reward positions and magnitudes.
    """
    assert corridor_width > 0, 'Corridor width must be positive!'
    # ensure that stem and arm are at least as long as the corridor is wide
    center_height = max(corridor_width, center_height)
    lap_width = max(corridor_width, lap_width)
    height = center_height + 2 * corridor_width
    width = lap_width * 2 + 3 * corridor_width
    # environmental borders
    borders_coords = np.array(
        [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
    )
    left_lap = np.array(
        [
            [corridor_width, corridor_width],
            [corridor_width + lap_width, corridor_width],
            [corridor_width + lap_width, height - corridor_width],
            [corridor_width, height - corridor_width],
            [corridor_width, corridor_width],
        ]
    )
    borders = sh.Polygon(borders_coords)
    borders = borders.difference(sh.Polygon(left_lap))
    borders = borders.difference(
        sh.Polygon(left_lap + np.array([lap_width + corridor_width, 0]))
    )
    # reward location
    rewards = np.array([])
    if goal_arm in ['left', 'right']:
        rewards = np.array(
            [
                [
                    corridor_width / 2
                    + (goal_arm == 'right') * (width - corridor_width),
                    height / 2,
                    reward,
                ]
            ]
        )
    # spawn area
    s_x, s_y = lap_width + corridor_width, (height - corridor_width) / 2
    spawn = np.array(
        [
            [s_x, s_y],
            [s_x + corridor_width, s_y],
            [s_x + corridor_width, s_y + corridor_width],
            [s_x, s_y + corridor_width],
            [s_x, s_y],
        ]
    )

    return borders, sh.Polygon(spawn), [], rewards


def make_cross_maze(
    arm_length: float,
    corridor_width: float,
    goal_arm: Literal['left', 'top', 'right', 'bottom'] = 'top',
    reward: float = 1,
    rotation: float = 0.0,
) -> ContinuousTemplate:
    """
    This function builds an cross environment.

    Parameters
    ----------
    arm_length : float
        The 8-maze's center length.
    corridor_width : float
        The 8-maze's corridor width.
    goal_arm : str, default='right'
        The rewarded lap (left, right, top, bottom, none).
    reward : float, default=1.
        The reward provided at the rewarded lap.
    rotation : float, default=0.
        The amount by which the environment should be rotated by in degrees.

    Returns
    -------
    room : sh.Polygon
        A polygon representing the environmental borders.
    spawn : sh.Polygon
        A polygon representing the agent's starting area.
    obstacles : list of sh.Polygon
        A list of polygons representing the environment's obstacles.
    rewards : NDArray
        A numpy array encoding the reward positions and magnitudes.
    """
    assert arm_length > 0, 'The arm length must be positive!'
    assert corridor_width > 0, 'Corridor width must be positive!'
    assert goal_arm in ('left', 'top', 'right', 'bottom'), 'Invalid goal arm!'
    w = corridor_width / 2  # half width
    l = w + arm_length  # center to arm end
    theta = np.deg2rad(rotation)
    R = np.array(  # noqa: N806
        [(np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))]
    )
    # environmental borders
    borders_coords = np.array(
        [
            (-l, w),
            (-w, w),
            (-w, l),
            (w, l),
            (w, w),
            (l, w),
            (l, -w),
            (w, -w),
            (w, -l),
            (-w, -l),
            (-w, -w),
            (-l, -w),
            (-l, w),
        ]
    )
    for i in range(borders_coords.shape[0]):
        borders_coords[i] = R @ borders_coords[i]
    borders = sh.Polygon(borders_coords)
    # reward location
    rewards: NDArray
    if goal_arm == 'left':
        rewards = np.array([(-arm_length, 0)])
    elif goal_arm == 'top':
        rewards = np.array([(0, arm_length)])
    elif goal_arm == 'right':
        rewards = np.array([(arm_length, 0)])
    elif goal_arm == 'bottom':
        rewards = np.array([(0, -arm_length)])
    rewards[0] = R @ rewards[0]
    rewards = np.hstack((rewards, np.full((1, 1), reward)))
    # spawn area
    spawn = np.array([(-w, w), (w, w), (w, -w), (-w, -w), (-w, w)])
    for i in range(spawn.shape[0]):
        spawn[i] = R @ spawn[i]

    return borders, sh.Polygon(spawn), [], rewards


def make_rectangle(
    location: NDArray, width: float, height: float, orientation: float = 0.0
) -> sh.Polygon:
    """
    This function builds a rectangular obstacle.

    Parameters
    ----------
    location : NDArray
        The location of the rectangular obstacle.
    width : float
        The rectangle's width.
    height : float
        The rectangle's height.
    orientation : float, default=0.
        The rectangle's orientation.

    Returns
    -------
    obstacle : sh.Polygon
        A polygon representing the rectangular obstacle.
    """
    h, w = height / 2, width / 2
    obstacle = sh.Polygon(np.array([[-w, -h], [w, -h], [w, h], [-w, h], [-w, -h]]))
    obstacle = rotate(obstacle, orientation, (0, 0))
    obstacle = translate(obstacle, location[0], location[1])

    return obstacle


def make_circle(location: NDArray, radius: float) -> sh.Polygon:
    """
    This function builds a circular obstacle.

    Parameters
    ----------
    location : NDArray
        The location of the circular obstacle.
    radius : float
        The circle's radius.

    Returns
    -------
    obstacle : sh.Polygon
        A polygon representing the rectangular obstacle.
    """
    obstacle = sh.Point(location)
    obstacle = obstacle.buffer(radius)

    return obstacle


def make_triangle(
    location: NDArray,
    width: float,
    height: float,
    base: float = 0.5,
    orientation: float = 0.0,
) -> sh.Polygon:
    """
    This function builds a triangular obstacle.

    Parameters
    ----------
    location : NDArray
        The location of the triangle with points ABC.
    width : float
        The length of the triangle's base AB.
    height : float
        The height of the triangle.
    base : float, default=0.5
        The position along the base AB which is closest to C.
    orientation : float, default=0.
        The triangle's orientation.

    Returns
    -------
    obstacle : sh.Polygon
        A polygon representing the rectangular obstacle.
    """
    obstacle = sh.Polygon(
        np.array([[0, 0], [width, 0], [width * base, height], [0, 0]])
    )
    center_x, center_y = list(obstacle.centroid.coords)[0]
    obstacle = translate(obstacle, location[0] - center_x, location[1] - center_y)
    obstacle = rotate(obstacle, orientation)

    return obstacle
