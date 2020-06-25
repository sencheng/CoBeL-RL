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

class UnityPerformanceMonitor(): 
        """
        QtWindow for basic plotting
        """

        def __init__(self, update_rate=None, calculation_range=100, reward_plot_viewbox=(-1,1,20), steps_plot_viewbox=(0,5000,20)):
            """
            Constructor 
            :param update_rate: number of steps to pass before updating the plot
            :nb_trace_length: defines how many episodes will be shown in the plot
            """

            # set update params
            self.update_rate = update_rate

            #set image axis mode
            pg.setConfigOptions(imageAxisOrder='row-major')

            # qt window
            self.win = pg.GraphicsWindow()
            self.win.resize(1000,800)
            self.win.setWindowTitle("Unity Environment Plot")

            # layout
            self.layout = pg.GraphicsLayout(border=(10,10,10))
            self.win.setCentralItem(self.layout)

            # add labels
            self.sps_label = self.layout.addLabel("")
            self.total_steps_label = self.layout.addLabel("")
            self.layout.nextRow()

            self.update_time_label = self.layout.addLabel("")
            self.memory_usage_label = self.layout.addLabel("")
            self.layout.nextRow()

            # pens
            self.raw_pen = pg.mkPen((255,255,255), width=1)
            self.mean_pen = pg.mkPen((255,0,0), width=2)
            self.var_pen = pg.mkPen((0,255,0), width=2)

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
            self.reward_plot_item = self.layout.addPlot(title="reward", viewBox=self.reward_plot_viewbox)
            self.reward_plot_item.showGrid(x=True, y=True)
            self.reward_graph = self.reward_plot_item.plot()
            self.mean_reward_graph = self.reward_plot_item.plot()
           
            self.steps_plot_item = self.layout.addPlot(title="steps per episode", viewBox=self.steps_plot_viewbox)
            self.steps_plot_item.showGrid(x=True, y= True)
            self.steps_graph = self.steps_plot_item.plot()
            self.mean_steps_graph = self.steps_plot_item.plot()

            # the episode range for calculating means and variances
            self.calculation_range = calculation_range

            # the observation plot will be initialized on receiving the
            # first observation and will adapt to the type of data given
            self.observation_plot_initialized = False

            # data traces
            self.reward_trace = []
            self.nb_episode_steps_trace = []
            self.mean_rewards_trace = []
            self.mean_nb_episode_steps_trace = []
            self.reward_variance_trace = []
            self.nb_episode_steps_variance_trace = []

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
            elif self.update_rate is None:

                return
            
            # update after update_rate steps
            elif nb_step % self.update_rate == 0:

                t1 = time.perf_counter()

                pg.mkQApp().processEvents()

                self.update_time_label.setText("Plotting time: " + str(round(time.perf_counter() - t1, 3)))

        def set_step_data(self, nb_step, obs):
            """
            update the labels and the observation
            this is very fast and is done every step
            :param nb_step: number of the current learning step
            :param obs: the current observation
            """
            self.set_sps(int(nb_step/(time.perf_counter() - self.start_time)))
            self.set_nb_steps(nb_step) 
            self.set_obs(obs) 

        def set_episode_data(self, nb_episode, nb_episode_steps, cumulativ_reward):
            """
            update the episode data plots
            this is rather slow and is just done on the end of an episode
            :param nb_episode: number of the current episode
            :param nb_episode_steps: the number of steps of this episode
            :param cumulativ_reward: the reward for this episode
            """
            self.set_episode_plot(nb_episode, nb_episode_steps, cumulativ_reward)


        def set_sps(self, sps):
            """
            set the sps value to the label
            :param sps: steps per second
            """
            self.sps_label.setText("steps per second: " + str(sps))

        def set_nb_steps(self, steps):
            """
            set the number of steps to the label
            :param steps: total number of steps
            """
            self.total_steps_label.setText("elapsed steps: " + str(steps))

        def set_episode_plot(self, nb_episode, nb_episode_steps, reward):
            """
            calculates the mean and the variance and plots the values in the corresponding graphs.
            :param nb_episode: number of the current episode
            :param nb_episode_steps: the nubmer of steps of this episode
            :param reward: the reward for this episode.
            """
            # append data
            self.reward_trace.append(reward)
            self.nb_episode_steps_trace.append(nb_episode_steps)

            # slices for mean calculation
            reward_slice = []
            steps_slice = []

            # get the slices for mean calculation
            # if the mean calculation range exceeds the number of gathered value
            # use all existing data as slices
            # else get slices of mean calculation size.
            if nb_episode < self.calculation_range:
                reward_slice = self.reward_trace
                steps_slice  = self.nb_episode_steps_trace
            else:
                reward_slice = self.reward_trace[-self.calculation_range:]
                steps_slice  = self.nb_episode_steps_trace[-self.calculation_range:]

            # calculate the means
            mean_reward = np.mean(reward_slice)
            mean_steps = np.mean(steps_slice)

            # calculate variances
            var_reward = np.var(reward_slice)
            var_steps  = np.var(steps_slice)

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

            self.memory_usage_label.setText("Plot data: " + str((sys.getsizeof(self.reward_trace) 
                                                + sys.getsizeof(self.mean_rewards_trace) 
                                                + sys.getsizeof(self.nb_episode_steps_trace) 
                                                + sys.getsizeof(self.mean_nb_episode_steps_trace)
                                                + sys.getsizeof(self.reward_variance_trace)
                                                + sys.getsizeof(self.nb_episode_steps_variance_trace))/1000) + " MB")
            
        def set_obs(self, observation):
            """
            displays the given observation
            :param observation: observation to display
            """

            # if no observation was plotted before
            if not self.observation_plot_initialized:
                
                print("observation shape:", observation.shape)
                self.layout.nextRow()

                # analyse data dimensions
                if len(observation.shape) == 1:
                    
                    # plot as vector
                    self.obs_plt = self.layout.addPlot(title="Vector observation", colspan=2)

                elif len(observation.shape) >= 3:

                    # plot as image
                    self.imvb = self.layout.addViewBox(lockAspect=True, colspan=2, enableMouse=False)
                    self.img = pg.ImageItem(observation)
                    self.imvb.addItem(self.img)

                self.observation_plot_initialized = True

            # plot as image
            if len(observation.shape) >= 3:
                self.img.setImage(observation[::-1])
                self.img.setLevels([0,1])

            # plot as vector
            elif len(observation.shape) == 1:                
                self.obs_plt.plot(observation, clear=True)
                
        
class RLPerformanceMonitorBaseline():
    def __init__(self,rlAgent,guiParent,visualOutput):
        
        # store the rlAgent
        self.rlAgent=rlAgent
        self.guiParent=guiParent
        
        # shall visual output be provided?
        self.visualOutput=visualOutput
        
        #define the variables that will be monitored
        self.rlRewardTraceRaw=np.zeros(rlAgent.trialNumber,dtype='float')
        self.rlRewardTraceRefined=np.zeros(rlAgent.trialNumber,dtype='float')
        
        
        
        # this is the accumulation range for smoothing the reward curve
        self.accumulationRangeReward=20
        
        # this is the accumulation interval for correct/incorrect decisions at the beginning/end of the single experimental phases (acquisition,extinction,renewal) 
        self.accumulationIntervalPerformance=10
        
        if visualOutput:
            # redefine the gui's dimensions
            self.guiParent.setGeometry(0,0,1920,600)
            
            # set up the required plots
            self.rlRewardPlot = self.guiParent.addPlot( title="Reinforcement learning progress" )
            
            # set x/y-ranges for the plots
            self.rlRewardPlot.setXRange(0,rlAgent.trialNumber)
            self.rlRewardPlot.setYRange(-100.0,100.0)
            
            # define the episodes domain
            self.episodesDomain=np.linspace(0,rlAgent.trialNumber,rlAgent.trialNumber)
            
            
            # each variable has a dedicated graph that can be used for displaying the monitored values
            self.rlRewardTraceRawGraph=self.rlRewardPlot.plot(self.episodesDomain,self.rlRewardTraceRaw)
            self.rlRewardTraceRefinedGraph=self.rlRewardPlot.plot(self.episodesDomain,self.rlRewardTraceRefined)

            
            
        
        
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
    def update(self,trial,logs):
        print('update')
        # update the reward traces
        rlReward=logs['episode_reward']
        self.rlRewardTraceRaw[trial]=rlReward
        
        aggregatedRewardTraceRaw=None
        
        if trial<self.accumulationRangeReward:
            aggregatedRewardTraceRaw=self.rlRewardTraceRaw[trial:None:-1]
        else:
            aggregatedRewardTraceRaw=self.rlRewardTraceRaw[trial:trial-self.accumulationRangeReward:-1]
        self.rlRewardTraceRefined[trial]=np.mean(aggregatedRewardTraceRaw)
        
        
        if self.visualOutput:
            
            # set the graph's data
            self.rlRewardTraceRawGraph.setData(self.episodesDomain,self.rlRewardTraceRaw,pen=pg.mkPen(color=(128,128,128),width=1))
            self.rlRewardTraceRefinedGraph.setData(self.episodesDomain,self.rlRewardTraceRefined,pen=pg.mkPen(color=(255,0,0),width=2))
        
        
                    
