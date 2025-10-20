# basic imports
import copy
from itertools import product
import numpy as np
import shapely as sh  # type: ignore
# typing
from typing import Literal
from ..interface.topology import Node, NodeID

LimitsTuple = tuple[float, float]


def linear_track(
    nb_nodes_track: int,
    nb_nodes_width: int,
    spacing: float = 1.0,
    reward: float = 1.0,
    location: Literal['left', 'right'] = 'right',
) -> tuple[dict[NodeID, Node], list[NodeID]]:
    """
    This function generates topology nodes for a linear track environment.

    Parameters
    ----------
    nb_nodes_track : int
        The length of the track in nodes.
    nb_nodes_width : int
        The width of the track in nodes.
    spacing : float, default=1.
        The spacing between nodes.
    reward : float, default=1.
        The reward received when the end of the track is reached.
    location : str, default='right'
        The reward location, i.e., left or right.
        The starting state will be located on the opposite side.

    Returns
    -------
    nodes : dict of Node
        The topology nodes.
    starting_nodes : list of NodeID
        The starting nodes.
    """
    assert nb_nodes_track > 1, 'Track has to be at least 2 states long!'
    assert nb_nodes_width > 0, 'Track has to be at least 1 state wide!'
    assert spacing > 0, 'Node spacing must be positive!'
    assert location in ['left', 'right'], 'Invalid reward location!'
    # compute coordinates
    coordinates = np.array(
        [
            [i, nb_nodes_width - j - 1]
            for j in range(nb_nodes_width)
            for i in range(nb_nodes_track)
        ]
    ).astype(float)
    # generate topology nodes
    nodes: dict[NodeID, Node] = {}
    for n, (x, y) in enumerate(coordinates):
        node_id, pose = str(n), (x, y, 0.0, 0.0, 0.0, 0.0)
        neighbors = [node_id] * 4
        nodes[node_id] = {
            'id': node_id,
            'pose': pose,
            'terminal': False,
            'reward': 0.0,
            'neighbors': neighbors,
        }
    # determine neighbors
    for id_1, node_1 in nodes.items():
        for id_2, node_2 in nodes.items():
            dist = np.array(node_1['pose'][:2]) - np.array(node_2['pose'][:2])
            if id_1 != id_2:
                if 0 < dist[0] < 1.5 and abs(dist[1]) < 1:
                    node_1['neighbors'][0] = id_2  # left
                if 0 > dist[1] > -1.5 and abs(dist[0]) < 1:
                    node_1['neighbors'][1] = id_2  # up
                if 0 > dist[0] > -1.5 and abs(dist[1]) < 1:
                    node_1['neighbors'][2] = id_2  # right
                if 0 < dist[1] < 1.5 and abs(dist[0]) < 1:
                    node_1['neighbors'][3] = id_2  # down
    # adjust spacing
    for _, node in nodes.items():
        node['pose'] = (
            node['pose'][0] * spacing,
            node['pose'][1] * spacing,
            node['pose'][2],
            node['pose'][3],
            node['pose'][4],
            node['pose'][5],
        )
    # set goal and starting nodes
    starting_nodes: list[NodeID] = []
    for i in range(nb_nodes_width):
        node_id = str(i * nb_nodes_track + (nb_nodes_track - 1) * (location == 'right'))
        nodes[node_id].update({'terminal': True, 'reward': reward})
        node_id = str(i * nb_nodes_track + (nb_nodes_track - 1) * (location == 'left'))
        starting_nodes.append(node_id)

    return nodes, starting_nodes


def grid(
    nb_nodes: int | tuple[int, int],
    limits: LimitsTuple | tuple[LimitsTuple, LimitsTuple] = (0.0, 1.0),
    reward: float = 1.0,
    location: None | NodeID = None,
) -> tuple[dict[NodeID, Node], list[NodeID]]:
    """
    This function generates topology nodes for a grid graph environment.

    Parameters
    ----------
    nb_nodes : int or 2-tuple of int
        The number of nodes per side.
        Can be int (same number for both sides) or tuple.
    limits : LimitsTuple or 2-tuple of LimitsTuple, default=(0., 1.)
        The coordinate ranges.
        Can be a simple tuple (same limits fo both sides) or tuple of tuples.
    reward : float, default=1.
        The reward received when the end of the goal node is reached.
    location : NodeID or None, optional
        The goal node. If none or invalid the top right node
        becomes the goal node. All other nodes become possible starting nodes.

    Returns
    -------
    nodes : dict of Node
        The topology nodes.
    starting_nodes : list of NodeID
        The starting nodes.
    """
    # prepare nodes and environment limits, and validate them
    nb_nodes_x = nb_nodes if (isinstance(nb_nodes, int)) else nb_nodes[0]
    nb_nodes_y = nb_nodes if (isinstance(nb_nodes, int)) else nb_nodes[1]
    assert (nb_nodes_x > 1 and nb_nodes_y >= 1) or (
        nb_nodes_x >= 1 and nb_nodes_y > 1
    ), 'Invalid environment dimensions!'
    limits_x = limits if (isinstance(limits[0], float)) else limits[0]
    assert limits_x[1] > limits_x[0], 'Invalid x coordinate range!'
    limits_y = limits if (isinstance(limits[0], float)) else limits[1]
    assert limits_y[1] > limits_y[0], 'Invalid y coordinate range!'
    # compute node coordinates
    coordinates_x = np.linspace(limits_x[0], limits_x[1], nb_nodes_x)
    coordinates_y = np.linspace(limits_y[0], limits_y[1], nb_nodes_y)
    # generate toplogy nodes
    nodes: dict[NodeID, Node] = {}
    for n, (y, x) in enumerate(product(coordinates_y, coordinates_x)):
        node_id = str(n)
        pose = (float(x), float(limits_y[1] - (y - limits_y[0])), 0.0, 0.0, 0.0, 0.0)
        j, i = divmod(n, nb_nodes_x)
        neighbors: list[NodeID] = []
        neighbors.append(str(int(j * nb_nodes_x + max(i - 1, 0))))  # left
        neighbors.append(str(int(max(j - 1, 0) * nb_nodes_x + i)))  # up
        neighbors.append(str(int(j * nb_nodes_x + min(i + 1, nb_nodes_x - 1))))  # right
        neighbors.append(str(int(min(j + 1, nb_nodes_y - 1) * nb_nodes_x + i)))  # down
        nodes[node_id] = {
            'id': node_id,
            'pose': pose,
            'terminal': False,
            'reward': 0.0,
            'neighbors': neighbors,
        }
    # set goal node
    if location is None or location not in nodes:
        location = str(nb_nodes_x - 1)
    nodes[location].update({'terminal': True, 'reward': reward})
    # prepare starting nodes
    starting_nodes: list[NodeID] = list(nodes.keys())
    starting_nodes.remove(location)

    return nodes, starting_nodes


def hexagonal(
    nb_nodes: int,
    limits: LimitsTuple = (0.0, 1.0),
    reward: float = 1.0,
    location: None | NodeID = None,
) -> tuple[dict[NodeID, Node], list[NodeID]]:
    """
    This function generates topology nodes for a hexagonal graph environment.

    Parameters
    ----------
    nb_nodes : int
        The number of nodes along the vertical side.
    limits : LimitsTuple, default=(0., 1.)
        The coordinate ranges.
    reward : float, default=1.
        The reward received when the end of the goal node is reached.
    location : NodeID or None, optional
        The goal node. If none or invalid the top right node
        becomes the goal node. All other nodes become possible starting nodes.

    Returns
    -------
    nodes : dict of Node
        The topology nodes.
    starting_nodes : list of NodeID
        The starting nodes.
    """
    assert nb_nodes > 1, 'Invalid number of nodes!'
    assert limits[1] > limits[0], 'Invalid coordinate range!'
    # compute node coordinates and spacing
    spacing = (limits[1] - limits[0]) / (nb_nodes - 1)
    coordinates = np.array(
        [
            [x, y]
            for y, x in product(
                np.linspace(limits[0], limits[1], nb_nodes),
                np.linspace(limits[0], limits[1], nb_nodes),
            )
        ]
    )
    idx = []
    for i in range(nb_nodes):
        if i % 2 == 1:
            idx += [i * nb_nodes + j for j in range(nb_nodes)]
    coordinates[idx, 0] += spacing / 2
    coordinates = coordinates[coordinates[:, 0] <= limits[1]]
    # generate topology nodes
    nodes: dict[NodeID, Node] = {}
    for n, (x, y) in enumerate(coordinates):
        node_id, pose = str(n), (x, limits[1] - (y - limits[0]), 0.0, 0.0, 0.0, 0.0)
        neighbors: list[NodeID] = [node_id] * 6
        nodes[node_id] = {
            'id': node_id,
            'pose': pose,
            'terminal': False,
            'reward': 0.0,
            'neighbors': neighbors,
        }
    # determine neighbors
    for id_1, node_1 in nodes.items():
        for id_2, node_2 in nodes.items():
            dist = np.sqrt(
                np.sum(
                    (np.array(node_1['pose'][:2]) - np.array(node_2['pose'][:2])) ** 2
                )
            )
            if id_1 != id_2 and dist < spacing * 1.5:
                node_1['neighbors'].append(id_2)
    # sort neighbors (from left, i.e., 270 deg, clockwise)
    for node_id, node in nodes.items():
        neighbors_sorted = {(i * 60 - 120) % 360: node_id for i in range(6)}
        angles = np.array([angle for angle in neighbors_sorted])
        pose_current = np.array(node['pose'][:2])
        for neighbor in node['neighbors']:
            pose_neighbor = np.array(nodes[neighbor]['pose'][:2])
            angle = (
                np.angle(
                    complex(
                        pose_neighbor[0] - pose_current[0],
                        pose_neighbor[1] - pose_current[1],
                    ),
                    deg=True,
                )
                % 360
            )
            key_angle = angles[int(np.argmin(np.abs(angles - angle)))]
            neighbors_sorted[key_angle] = neighbor
        node['neighbors'] = list(neighbors_sorted.values())[::-1]
    # set goal node
    if location is None or location not in nodes:
        location = str(nb_nodes - 1)
    nodes[location].update({'terminal': True, 'reward': reward})
    # prepare starting nodes
    starting_nodes: list[NodeID] = list(nodes.keys())
    starting_nodes.remove(location)

    return nodes, starting_nodes


def t_maze(
    nb_nodes_stem: int,
    nb_nodes_arm: int,
    nb_nodes_width: int,
    spacing: float = 1.0,
    reward: float = 1.0,
    location: Literal['left', 'right'] = 'right',
) -> tuple[dict[NodeID, Node], list[NodeID]]:
    """
    This function generates topology nodes for a hexagonal graph environment.

    Parameters
    ----------
    nb_nodes_stem : int
        The length of the stem in nodes.
    nb_nodes_arm : int
        The length of each arm in nodes.
    nb_nodes_width : int
        The width of the corridor in nodes.
    spacing : float, default=1.
        The spacing between nodes.
    reward : float, default=1.
        The reward received when the end of the goal node is reached.
    location : str, default='right'
        The goal node. If none or invalid the top right node
        becomes the goal node. All other nodes become possible starting nodes.

    Returns
    -------
    nodes : dict of Node
        The topology nodes.
    starting_nodes : list of NodeID
        The starting nodes.
    """
    assert nb_nodes_arm > 0, 'Invalid arm length!'
    assert nb_nodes_stem > 0, 'Invalid stem length!'
    assert nb_nodes_width > 0, 'Invalid corridor width!'
    assert location in ['left', 'right'], 'The goal can only be located left or right!'
    # compute node coordiantes
    coordinates = []
    arm_span = nb_nodes_arm * 2 + nb_nodes_width
    for i in range(nb_nodes_width):
        for j in range(arm_span):
            x = j
            y = nb_nodes_stem + nb_nodes_width - 1 - i
            coordinates.append([x, y])
    for i in range(nb_nodes_stem):
        for j in range(nb_nodes_width):
            x = nb_nodes_arm + j
            y = nb_nodes_stem - 1 - i
            coordinates.append([x, y])
    # generate topology nodes
    nodes: dict[NodeID, Node] = {}
    for n, (x, y) in enumerate(coordinates):
        node_id, pose = str(n), (x, y, 0.0, 0.0, 0.0, 0.0)
        neighbors: list[NodeID] = [node_id] * 4
        nodes[node_id] = {
            'id': node_id,
            'pose': pose,
            'terminal': False,
            'reward': 0.0,
            'neighbors': neighbors,
        }
    # determine neighbors
    for id_1, node_1 in nodes.items():
        for id_2, node_2 in nodes.items():
            dist = np.array(node_1['pose'][:2]) - np.array(node_2['pose'][:2])
            if id_1 != id_2:
                if 0 < dist[0] < 1.5 and abs(dist[1]) < 1:
                    node_1['neighbors'][0] = id_2  # left
                if 0 > dist[1] > -1.5 and abs(dist[0]) < 1:
                    node_1['neighbors'][1] = id_2  # up
                if 0 > dist[0] > -1.5 and abs(dist[1]) < 1:
                    node_1['neighbors'][2] = id_2  # right
                if 0 < dist[1] < 1.5 and abs(dist[0]) < 1:
                    node_1['neighbors'][3] = id_2  # down
    # adjust spacing
    for _, node in nodes.items():
        node['pose'] = (
            node['pose'][0] * spacing,
            node['pose'][1] * spacing,
            node['pose'][2],
            node['pose'][3],
            node['pose'][4],
            node['pose'][5],
        )
    # set goal node(s)
    for i in range(nb_nodes_width):
        node_id = str(
            (nb_nodes_arm * 2 + nb_nodes_width) * i
            + (nb_nodes_arm * 2 + nb_nodes_width - 1) * int(location == 'right')
        )
        nodes[node_id].update({'terminal': True, 'reward': reward})
    # prepare starting node(s)
    starting_nodes: list[NodeID] = []
    for node_id in list(nodes.keys())[-nb_nodes_width:]:
        starting_nodes.append(node_id)

    return nodes, starting_nodes


def cross(
    nb_nodes_arm: int, nb_nodes_width: int, spacing: float = 1.0, rotation: float = 0.0
) -> tuple[dict[NodeID, Node], list[NodeID]]:
    """
    This function generates topology nodes for a cross arena environment.

    Parameters
    ----------
    nb_nodes_arm : int
        The length of each cross "arm" in nodes.
    nb_nodes_width : int
        The width of the corridor in nodes.
    spacing : float, default=1.
        The spacing between nodes.
    rotation : float, default=0.
        The amount by which the environment should be rotated by in degrees.

    Returns
    -------
    nodes : dict of Node
        The topology nodes.
    starting_nodes : list of NodeID
        The starting nodes.
    """
    assert nb_nodes_arm > 0, 'The arm must be at least 1 node long!'
    assert nb_nodes_width > 0, 'The corridors must be at least 1 node wide!'
    assert spacing > 0, 'Node spacing must be positive!'
    # compute coordinates
    nb_nodes_diameter = 2 * nb_nodes_arm + nb_nodes_width
    ticks = np.linspace(0, 1.0, nb_nodes_diameter)
    coord = []
    # upper part
    for i in range(nb_nodes_arm):
        for j in range(nb_nodes_width):
            coord.append((ticks[nb_nodes_arm + j], ticks[-(1 + i)]))
    # center part
    for j in range(nb_nodes_width):
        for i in range(nb_nodes_diameter):
            coord.append((ticks[i], ticks[-(nb_nodes_arm + 1 + j)]))
    # lower part
    for i in range(nb_nodes_arm):
        for j in range(nb_nodes_width):
            coord.append(
                (
                    ticks[nb_nodes_arm + j],
                    ticks[-(i + 1 + nb_nodes_arm + nb_nodes_width)],
                )
            )
    # generate nodes
    nodes: dict[NodeID, Node] = {}
    for n, (x, y) in enumerate(coord):
        node_id, pose = str(n), (x, y, 0.0, 0.0, 0.0, 0.0)
        neighbors = [node_id] * 4
        nodes[node_id] = {
            'id': node_id,
            'pose': pose,
            'terminal': False,
            'reward': 0.0,
            'neighbors': neighbors,
        }
    # determine neighbors
    t = 1.1 / (nb_nodes_diameter - 1.0)
    for id_1, node_1 in nodes.items():
        for id_2, node_2 in nodes.items():
            dist = np.array(node_1['pose'][:2] - np.array(node_2['pose'][:2]))
            if id_1 != id_2:
                if 0 < dist[0] < t and abs(dist[1]) < t / 2:
                    node_1['neighbors'][0] = id_2
                if 0 > dist[1] > -t and abs(dist[0]) < t / 2:
                    node_1['neighbors'][1] = id_2
                if 0 > dist[0] > -t and abs(dist[1]) < t / 2:
                    node_1['neighbors'][2] = id_2
                if 0 < dist[1] < t and abs(dist[0]) < t / 2:
                    node_1['neighbors'][3] = id_2
    # adjust spacing, center and rotate cross
    s = (nb_nodes_diameter - 1) * spacing
    offset = spacing * (nb_nodes_diameter - 1) / 2
    theta = np.deg2rad(rotation)
    R = np.array(  # noqa: N806
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    for _, node in nodes.items():
        x, y = R @ (np.array(node['pose'][:2]) * s - offset)
        node['pose'] = (
            float(x),
            float(y),
            node['pose'][2],
            node['pose'][3],
            node['pose'][4],
            node['pose'][5],
        )
    starting_nodes: list[NodeID] = list(nodes.keys())

    return nodes, starting_nodes


def remove_obstructed_neighbors(
    nodes: dict[NodeID, Node], obstacles: list[sh.Polygon], buffer_distance: float = 0.0
) -> dict[NodeID, Node]:
    """
    This function removes the edges of a provided topology graph
    which are obstructed by a given set of obstacles.
    Note: Due to the use of shapely this function ignores
    the z-coordinate (height) of nodes and obstacles.

    Parameters
    ----------
    nodes : dict of Node
        A dictionary containing the topology graph nodes.
    obstacles : list of sh.Polygon
        A list of polygons representing environmental obstacles.
    buffer_distance : float, default=0.
        Additional buffer distance applied to the obstacles.

    Returns
    -------
    nodes_updated : dict of Node
        A dictionary containg the updated topology graph nodes.
    """
    assert buffer_distance >= 0, 'The buffer distance has to be non-negative!'
    nodes_updated = copy.deepcopy(nodes)
    obs = sh.buffer(sh.MultiPolygon(obstacles), buffer_distance)
    for n, node in nodes_updated.items():
        pos_1 = np.array(node['pose'])[:2]
        for i, neighbor in enumerate(node['neighbors']):
            pos_2 = np.array(nodes_updated[neighbor]['pose'][:2])
            if sh.intersects(sh.LineString((pos_1, pos_2)), obs):
                node['neighbors'][i] = n

    return nodes_updated
