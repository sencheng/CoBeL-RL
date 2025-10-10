# basic imports
import numpy as np
import pyqtgraph as pg  # type: ignore
# framework imports
from .monitor import Monitor
# typing
from numpy.typing import NDArray
from ..agent.agent import Logs
from ..interface.interface import Interface
from .monitor import Trace
from ..network.network import Batch


class EscapeLatencyMonitor(Monitor):
    """
    This class implements a monitor which tracks the number of steps per trial.

    Parameters
    ----------
    trials : int
        The number of trials that will be tracked.
    max_steps : int
        The maximum number of steps per trial. Used for visualization.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    max_steps : int
        The maximum number of steps per trial. Used for visualization.
    latency_trace : list of NDArray
        The escape latency trace.
    latency_trace_avg : list of NDArray
        The smoothed (average over the 10 most recent trials) escape latency trace.

    Examples
    --------

    Setting up the monitor to track escape latency
    (callbacks are assumed to have been passed to the agent). ::

        >>> from cobel.monitor import EscapeLatencyMonitor
        >>> monitor = EscapeLatencyMonitor(100, 50)
        >>> callbacks = {'on_trial_end': [monitor.update]}

    """

    def __init__(
        self, trials: int, max_steps: int, widget: None | pg.GraphicsLayoutWidget = None
    ) -> None:
        super().__init__(widget)
        self.max_steps = max_steps
        self.latency_trace = np.full(trials, float('nan'), dtype='float')
        self.latency_trace_avg = np.copy(self.latency_trace)
        if self.widget is not None:
            self.widget.setGeometry(50, 50, 1600, 600)
            self.el_plot = self.widget.addPlot(title='Escape Latency')
            self.el_plot.setXRange(0, trials)
            self.el_plot.setYRange(0, max_steps * 1.05)
            self.trials = np.linspace(0, trials, trials)
            self.el_graph = self.el_plot.plot(self.trials, self.latency_trace)
            self.el_graph_avg = self.el_plot.plot(self.trials, self.latency_trace_avg)
            self.refresh_visualization()

    def clear_plots(self) -> None:
        """
        This function clears all plots.
        """
        if self.widget is not None:
            self.widget.removeItem(self.el_plot)

    def update(self, logs: Logs) -> None:
        """
        This function updates the monitor.

        Parameters
        ----------
        logs : Logs
            The log dictionary.
        """
        trial = logs['trial']
        self.latency_trace[trial] = logs['steps']
        avg = np.nanmean(self.latency_trace[max(0, trial - 10) : (trial + 1)])
        self.latency_trace_avg[trial] = self.max_steps if np.isnan(avg) else avg
        if self.widget is not None:
            self.el_graph.setData(
                self.trials,
                self.latency_trace,
                pen=pg.mkPen(color=(128, 128, 128), width=1),
            )
            self.el_graph_avg.setData(
                self.trials,
                self.latency_trace_avg,
                pen=pg.mkPen(color=(255, 0, 0), width=2),
            )
            self.refresh_visualization()

    def get_trace(self) -> Trace:
        """
        This function returns the escape latency trace.

        Returns
        -------
        trace : Trace
            The trace of the monitored variable.
        """
        return self.latency_trace


class RewardMonitor(Monitor):
    """
    This class implements a monitor which tracks the cumulative episodic rewards.

    Parameters
    ----------
    trials : int
        The number of trials that will be tracked.
    reward_range : 2-tuple of float, default=(0., 1.)
        The range of rewards. Used for visualization.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    reward_range : 2-tuple of float
        The range of rewards. Used for visualization.
    reward_trace : list of NDArray
        The reward trace.
    reward_trace_avg : list of NDArray
        The smoothed (average over the 10 most recent trials) reward trace.

    Examples
    --------

    Setting up the monitor to track the episodic reward
    (callbacks are assumed to have been passed to the agent). ::

        >>> from cobel.monitor import RewardMonitor
        >>> monitor = RewardMonitor(100)
        >>> callbacks = {'on_trial_end': [monitor.update]}

    """

    def __init__(
        self,
        trials: int,
        reward_range: tuple[float, float] = (0.0, 1.0),
        widget: None | pg.GraphicsLayoutWidget = None,
    ) -> None:
        super().__init__(widget)
        assert reward_range[1] > reward_range[0], 'Invalid reward range!'
        self.reward_range = reward_range
        self.reward_trace = np.full(trials, float('nan'), dtype='float')
        self.reward_trace_avg = np.copy(self.reward_trace)
        if self.widget is not None:
            self.widget.setGeometry(50, 50, 1600, 600)
            self.reward_plot = self.widget.addPlot(title='Episodic Reward')
            self.reward_plot.setXRange(0, trials)
            self.reward_plot.setYRange(self.reward_range[0], self.reward_range[1])
            self.trials = np.linspace(0, trials, trials)
            self.reward_graph = self.reward_plot.plot(self.trials, self.reward_trace)
            self.reward_graph_avg = self.reward_plot.plot(
                self.trials, self.reward_trace_avg
            )
            self.refresh_visualization()

    def clear_plots(self) -> None:
        """
        This function clears all plots.
        """
        if self.widget is not None:
            self.widget.removeItem(self.reward_plot)

    def update(self, logs: Logs) -> None:
        """
        This function updates the monitor.

        Parameters
        ----------
        logs : Logs
            The log dictionary.
        """
        trial = logs['trial']
        self.reward_trace[trial] = logs['trial_reward']
        avg = np.nanmean(self.reward_trace[max(0, trial - 10) : (trial + 1)])
        self.reward_trace_avg[trial] = self.reward_range[0] if np.isnan(avg) else avg
        if self.widget is not None:
            self.reward_graph.setData(
                self.trials,
                self.reward_trace,
                pen=pg.mkPen(color=(128, 128, 128), width=1),
            )
            self.reward_graph_avg.setData(
                self.trials,
                self.reward_trace_avg,
                pen=pg.mkPen(color=(255, 0, 0), width=2),
            )
            self.refresh_visualization()

    def get_trace(self) -> Trace:
        """
        This function returns the reward trace.

        Returns
        -------
        trace : Trace
            The trace of the monitored variable.
        """
        return self.reward_trace


class ResponseMonitor(Monitor):
    """
    This class implements a monitor which tracks the agent's responses.
    Responses must be encoded by the user and added to the trial log.
    If no response was defined the monitor will encode
    whether reward was received (1) or not (0).

    Parameters
    ----------
    trials : int
        The number of trials that will be tracked.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    responses : NDArray
        The response trace.
    CRC : NDArray
        The cumulative response curve.

    Examples
    --------

    Setting up the monitor to track when an agent executes action 3
    (callbacks are assumed to have been passed to the agent). ::

        >>> from cobel.monitor import ResponseMonitor
        >>> monitor = RewardMonitor(100)
        >>> def encode(logs):
        ...     logs['response'] = logs['action'] == 3
        ...     return logs
        >>> callbacks = {'on_step_end': [encode, monitor.update]}

    """

    def __init__(
        self, trials: int, widget: None | pg.GraphicsLayoutWidget = None
    ) -> None:
        super().__init__(widget)
        self.responses = np.full(trials, float('nan'), dtype='float')
        self.CRC = np.copy(self.responses)
        if self.widget is not None:
            self.widget.setGeometry(50, 50, 1600, 600)
            self.CRC_plot = self.widget.addPlot(title='Cumulative Response')
            self.CRC_plot.setXRange(0, 0)
            self.CRC_plot.setYRange(0, 0)
            self.trials = np.linspace(0, trials, trials)
            self.CRC_graph = self.CRC_plot.plot(self.trials, self.CRC)
            self.refresh_visualization()

    def clear_plots(self) -> None:
        """
        This function clears all plots.
        """
        if self.widget is not None:
            self.widget.removeItem(self.CRC_plot)

    def update(self, logs: Logs) -> None:
        """
        This function updates the monitor.

        Parameters
        ----------
        logs : Logs
            The log dictionary.
        """
        trial = logs['trial']
        if 'response' in logs:
            self.responses[trial] = logs['response']
        else:
            self.responses[trial] = int(logs['trial_reward'] > 0)
        self.CRC[trial] = np.sum(self.responses[: (trial + 1)])
        if self.widget is not None:
            self.CRC_plot.setXRange(0, trial)
            self.CRC_plot.setYRange(np.nanmin(self.CRC), np.nanmax(self.CRC))
            self.CRC_graph.setData(
                self.trials, self.CRC, pen=pg.mkPen(color=(128, 128, 128), width=1)
            )
            self.refresh_visualization()

    def get_trace(self) -> Trace:
        """
        This function returns the response trace.

        Returns
        -------
        trace : Trace
            The trace of the monitored variable.
        """
        return self.responses


class TrajectoryMonitor(Monitor):
    """
    This class implements a monitor which tracks the agent's trajectory.

    Parameters
    ----------
    trials : int
        The number of trials that will be tracked.
    env : Interface
        The environment that the agent is interacting with.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    env : Interface
        The environment that the agent is interacting with.
    trajectory_trace : list of list
        The trajectory trace.
    current_trial : int or None
        Tracks the current trial.

    Examples
    --------

    Setting up the monitor to track position in a gridworld environment
    (callbacks are assumed to have been passed to the agent). ::

        >>> from cobel.monitor import TrajectoryMonitor
        >>> from cobel.interface import Gridworld
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> env = Gridworld(make_open_field(4, 4))
        >>> monitor = TrajectoryMonitor(100, env)
        >>> callbacks = {'on_step_end': [monitor.update]}

    """

    def __init__(
        self, trials: int, env: Interface, widget: None | pg.GraphicsLayoutWidget = None
    ) -> None:
        super().__init__(widget)
        self.env = env
        self.trajectory_trace: list[list[NDArray]] = []
        self.current_trial: None | int = None
        if self.widget is not None:
            # to be implemented
            pass

    def clear_plots(self) -> None:
        """
        This function clears all plots.
        """
        if self.widget is not None:
            # to be implemented
            pass

    def update(self, logs: Logs) -> None:
        """
        This function updates the monitor.

        Parameters
        ----------
        logs : Logs
            The log dictionary.
        """
        if self.current_trial != logs['trial_session']:
            self.current_trial = logs['trial_session']
            self.trajectory_trace.append([])
        self.trajectory_trace[-1].append(self.env.get_position())
        if self.widget is not None:
            # to be implemented
            pass

    def get_trace(self) -> Trace:
        """
        This function returns the trajectory trace.

        Returns
        -------
        trace : Trace
            The trace of the monitored variable.
        """
        return self.trajectory_trace


class QMonitor(Monitor):
    """
    This class implements a monitor which tracks the agent's Q-function.

    Parameters
    ----------
    trials : int
        The number of trials that will be tracked.
    observations : Batch
        A set of observations for which the Q-function is tracked.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the monitor will be visualized.

    Attributes
    ----------
    observations : Batch
        A set of observations for which the Q-function is tracked.
    q_trace : list of NDArray
        The Q-function trace.

    Examples
    --------

    Setting up the monitor to track the Q-function at each step
    (callbacks are assumed to have been passed to the agent). ::

        >>> from cobel.monitor import QMonitor
        >>> observations = np.arange(16) # we track gridworld states
        >>> monitor = QMonitor(100, observations)
        >>> callbacks = {'on_step_end': [monitor.update]}

    """

    def __init__(
        self,
        trials: int,
        observations: Batch,
        widget: None | pg.GraphicsLayoutWidget = None,
    ) -> None:
        super().__init__(widget)
        self.observations = observations
        self.q_trace: list[NDArray] = []
        if self.widget is not None:
            # to be implemented
            pass

    def clear_plots(self) -> None:
        """
        This function clears all plots.
        """
        if self.widget is not None:
            # to be implemented
            pass

    def update(self, logs: Logs) -> None:
        """
        This function updates the monitor.

        Parameters
        ----------
        logs : Logs
            The log dictionary.
        """
        self.q_trace.append(logs['agent'].predict_on_batch(self.observations))
        if self.widget is not None:
            # to be implemented
            pass

    def get_trace(self) -> Trace:
        """
        This function returns the Q-function trace.

        Returns
        -------
        trace : Trace
            The trace of the monitored variable.
        """
        return self.q_trace
