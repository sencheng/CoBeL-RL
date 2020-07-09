import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import time
from time import perf_counter


def measure_time_decorator(func):
    """
    measures the time of an function execution.
    """

    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        value = func(self, *args, **kwargs)
        print(func, "called. elapsed time:", time.perf_counter() - start_time)
        return value

    return wrapper


class UnityPerformanceMonitor:
    """
    QtWindow for basic plotting
    """

    def __init__(self, update_period=None, calculation_range=100, reward_plot_viewbox=(-1, 1, 50),
                 steps_plot_viewbox=(0, 1000, 50)):
        """
        Constructor
        :param update_period: number of steps to pass before updating the plot
        :nb_trace_length: defines how many episodes will be shown in the plot
        """

        # set update params
        self.update_period = update_period

        # set image axis mode
        pg.setConfigOptions(imageAxisOrder='row-major')

        # qt window
        self.win = pg.GraphicsWindow()
        self.win.resize(1000, 800)
        self.win.setWindowTitle("Unity Environment Plot")

        # layout
        self.layout = pg.GraphicsLayout(border=(10, 10, 10))
        self.win.setCentralItem(self.layout)

        # add labels
        self.sps_label = self.layout.addLabel()
        self.nb_episodes_label = self.layout.addLabel()
        self.total_steps_label = self.layout.addLabel()

        self.layout.nextRow()

        # pens
        self.raw_pen = pg.mkPen((255, 255, 255), width=1)
        self.mean_pen = pg.mkPen((255, 0, 0), width=2)
        self.var_pen = pg.mkPen((0, 255, 0), width=2)

        # viewbox
        self.reward_plot_viewbox = pg.ViewBox(parent=self.layout, enableMouse=True, enableMenu=False)
        self.reward_plot_viewbox.setYRange(min=reward_plot_viewbox[0], max=reward_plot_viewbox[1])
        self.reward_plot_viewbox.setXRange(min=0, max=reward_plot_viewbox[2])
        self.reward_plot_viewbox.setAutoPan(x=True, y=False)
        self.reward_plot_viewbox.enableAutoRange(x=True, y=False)
        self.reward_plot_viewbox.setLimits(xMin=0)

        self.steps_plot_viewbox = pg.ViewBox(parent=self.layout, enableMouse=True, enableMenu=False)
        self.steps_plot_viewbox.setYRange(min=steps_plot_viewbox[0], max=steps_plot_viewbox[1])
        self.steps_plot_viewbox.setXRange(min=0, max=steps_plot_viewbox[2])
        self.steps_plot_viewbox.setAutoPan(x=True, y=False)
        self.steps_plot_viewbox.enableAutoRange(x=True, y=False)
        self.steps_plot_viewbox.setLimits(xMin=0)

        # episode plots
        self.reward_plot_item = self.layout.addPlot(title="reward", viewBox=self.reward_plot_viewbox, colspan=3)
        self.reward_plot_item.showGrid(x=True, y=True)
        self.reward_graph = self.reward_plot_item.plot()
        self.mean_reward_graph = self.reward_plot_item.plot()

        self.layout.nextRow()

        self.steps_plot_item = self.layout.addPlot(title="steps per episode", viewBox=self.steps_plot_viewbox, colspan=3)
        self.steps_plot_item.showGrid(x=True, y=True)
        self.steps_graph = self.steps_plot_item.plot()
        self.mean_steps_graph = self.steps_plot_item.plot()

        self.layout.nextRow()

        self.action_plot_item = self.layout.addPlot(title="actions", colspan=3)
        self.action_plot_item.getViewBox().enableAutoRange(x=True, y=False)
        self.action_plot_item.getViewBox().setYRange(min=-2, max=2)
        self.action_plot_item.getViewBox().setXRange(min=-20, max=20)
        self.action_plot_item.showGrid(x=True, y=True)
        self.action_scatter_plot = pg.ScatterPlotItem()
        self.action_plot_item.addItem(self.action_scatter_plot)

        self.layout.nextRow()

        # the observation plots will be initialized on receiving the
        # observation shapes and will adapt to the type of data given
        self.observation_plots = []
        self.observation_plot_viewbox = pg.ViewBox(parent=self.layout, enableMouse=False, enableMenu=False)

        # the episode range for calculating means and variances
        self.calculation_range = calculation_range

        # data traces
        self.reward_trace = []
        self.nb_episode_steps_trace = []
        self.mean_rewards_trace = []
        self.mean_nb_episode_steps_trace = []
        self.reward_variance_trace = []
        self.nb_episode_steps_variance_trace = []

        # save start time for sps calculation
        self.start_time = time.perf_counter()

    def update(self, nb_step=None):
        """
        update is called in every step
        :param nb_step: the number of learning iterations. set none for instant update.
        """
        # instant update
        if nb_step is None:
            pg.mkQApp().processEvents()

        # never update
        elif self.update_period is None:
            return

        # update after update_rate steps
        elif nb_step % self.update_period == 0:
            pg.mkQApp().processEvents()

    def set_step_data(self, nb_step):
        """
        update the labels
        this is very fast and is done every step
        :param nb_step: number of the current learning step
        :return:
        """
        self.set_sps(int(nb_step / (time.perf_counter() - self.start_time)))
        self.set_nb_steps(nb_step)

    def set_episode_data(self, nb_episode, nb_episode_steps, cumulative_reward):
        """
        update the episode data plots
        this is rather slow and is just done on the end of an episode
        :param nb_episode: number of the current episode
        :param nb_episode_steps: the number of steps of this episode
        :param cumulative_reward: the reward for this episode
        :return:
        """
        self.nb_episodes_label.setText(f'Episode: {nb_episode}')
        self.set_episode_plot(nb_episode, nb_episode_steps, cumulative_reward)

    def set_sps(self, sps):
        """
        set the sps value to the label
        :param sps: steps per second
        :return:
        """
        self.sps_label.setText("steps per second: " + str(sps))

    def set_nb_steps(self, steps):
        """
        set the number of steps to the label
        :param steps: total number of steps
        :return:
        """
        self.total_steps_label.setText("elapsed steps: " + str(steps))

    def set_episode_plot(self, nb_episode, nb_episode_steps, episode_reward):
        """
        calculates the mean and the variance and plots the values in the corresponding graphs.
        :param nb_episode: number of the current episode
        :param nb_episode_steps: the number of steps of this episode
        :param episode_reward: the reward for this episode.
        :return:
        """
        # append data
        self.reward_trace.append(episode_reward)
        self.nb_episode_steps_trace.append(nb_episode_steps)

        # get the slices for mean calculation
        # if the mean calculation range exceeds the number of gathered value
        # use all existing data as slices
        # else get slices of mean calculation size.
        if nb_episode < self.calculation_range:
            reward_slice = self.reward_trace
            steps_slice = self.nb_episode_steps_trace
        else:
            reward_slice = self.reward_trace[-self.calculation_range:]
            steps_slice = self.nb_episode_steps_trace[-self.calculation_range:]

        # calculate the means
        mean_reward = np.mean(reward_slice)
        mean_steps = np.mean(steps_slice)

        # calculate variances
        var_reward = np.var(reward_slice)
        var_steps = np.var(steps_slice)

        # append the means
        self.mean_rewards_trace.append(mean_reward)
        self.mean_nb_episode_steps_trace.append(mean_steps)

        # append variances
        self.reward_variance_trace.append(var_reward)
        self.nb_episode_steps_variance_trace.append(var_steps)

        # plot series
        self.reward_graph.setData(self.reward_trace, pen=self.raw_pen)
        self.mean_reward_graph.setData(self.mean_rewards_trace, pen=self.mean_pen)

        self.steps_graph.setData(self.nb_episode_steps_trace, pen=self.raw_pen)
        self.mean_steps_graph.setData(self.mean_nb_episode_steps_trace, pen=self.mean_pen)

    def instantiate_observation_plots(self, observation_shapes):
        """
        Instantiates the observation plots
        :param observation_shapes: a list of sensor observations shapes
        :return:
        """

        # instantiate plot for each observation
        for i, shape in enumerate(observation_shapes):

            # analyse data dimensions
            if len(shape) == 1:

                # plot as vector
                vector_plot = self.layout.addPlot(title="vector observation " + str(i), colspan=3).plot()
                self.observation_plots.append(vector_plot)

            elif len(shape) == 3 or len(shape) == 2:

                # plot as image
                self.observation_plots.append(pg.ImageItem(np.zeros(shape=shape)))
                self.layout.addItem(self.observation_plot_viewbox, colspan=3)
                self.observation_plot_viewbox.setAspectLocked(lock=True)
                self.observation_plot_viewbox.addItem(self.observation_plots[i])

            print(f"observation plot {i} instantiated with shape {shape}")

            self.layout.nextRow()

    def display_actions(self, actions):
        self.action_scatter_plot.clear()
        self.action_scatter_plot.addPoints(x=range(len(actions[0])), y=actions[0])

    def display_observations(self, observations):
        """
        displays the given observation
        :param observations: observation to display
        :return:
        """

        # plot observations
        for i, observation in enumerate(observations):

            # plot as image
            if len(observation.shape) == 3:
                # mirror image
                self.observation_plots[i].setImage(observation[::-1])
                # set color levels
                self.observation_plots[i].setLevels([0, 1])

            # plot as vector
            elif len(observation.shape) == 1:
                self.observation_plots[i].setData(observation, clear=True)


class RLPerformanceMonitorBaseline:
    def __init__(self, rlAgent, guiParent, visualOutput):

        # store the rlAgent
        self.rlAgent = rlAgent
        self.guiParent = guiParent

        # shall visual output be provided?
        self.visualOutput = visualOutput

        # define the variables that will be monitored
        self.rlRewardTraceRaw = np.zeros(rlAgent.trialNumber, dtype='float')
        self.rlRewardTraceRefined = np.zeros(rlAgent.trialNumber, dtype='float')

        # this is the accumulation range for smoothing the reward curve
        self.accumulationRangeReward = 20

        # this is the accumulation interval for correct/incorrect decisions at the beginning/end of the single experimental phases (acquisition,extinction,renewal) 
        self.accumulationIntervalPerformance = 10

        if visualOutput:
            # redefine the gui's dimensions
            self.guiParent.setGeometry(0, 0, 1920, 600)

            # set up the required plots
            self.rlRewardPlot = self.guiParent.addPlot(title="Reinforcement learning progress")

            # set x/y-ranges for the plots
            self.rlRewardPlot.setXRange(0, rlAgent.trialNumber)
            self.rlRewardPlot.setYRange(-100.0, 100.0)

            # define the episodes domain
            self.episodesDomain = np.linspace(0, rlAgent.trialNumber, rlAgent.trialNumber)

            # each variable has a dedicated graph that can be used for displaying the monitored values
            self.rlRewardTraceRawGraph = self.rlRewardPlot.plot(self.episodesDomain, self.rlRewardTraceRaw)
            self.rlRewardTraceRefinedGraph = self.rlRewardPlot.plot(self.episodesDomain, self.rlRewardTraceRefined)

    '''
    This function clears the plots generated by the performance monitor.
    '''

    def clearPlots(self):
        if self.visualOutput:
            self.guiParent.removeItem(self.rlRewardPlot)

    '''
    This function is called when a trial ends. Here, information about the monitored variables is memorized, and the monitor graphs are updated.
    
    trial:  the actual trial number
    logs:   information from the reinforcement learning subsystem
    '''

    def update(self, trial, logs):
        # update the reward traces
        rlReward = logs['episode_reward']
        self.rlRewardTraceRaw[trial] = rlReward

        aggregatedRewardTraceRaw = None

        if trial < self.accumulationRangeReward:
            aggregatedRewardTraceRaw = self.rlRewardTraceRaw[trial:None:-1]
        else:
            aggregatedRewardTraceRaw = self.rlRewardTraceRaw[trial:trial - self.accumulationRangeReward:-1]
        self.rlRewardTraceRefined[trial] = np.mean(aggregatedRewardTraceRaw)

        if self.visualOutput:
            # set the graph's data
            self.rlRewardTraceRawGraph.setData(self.episodesDomain, self.rlRewardTraceRaw,
                                               pen=pg.mkPen(color=(128, 128, 128), width=1))
            self.rlRewardTraceRefinedGraph.setData(self.episodesDomain, self.rlRewardTraceRefined,
                                                   pen=pg.mkPen(color=(255, 0, 0), width=2))
