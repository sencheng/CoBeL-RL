## **CoBeL-RL** ##
#### Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks ####

This is the newest version of CoBeL-RL.
If you require an older version, download the appropriately tagged commit/branch.

------------------------------------------------------
**About**

CoBeL-RL is a closed-loop simulator of complex behavior and learning based on RL and deep neural networks. It provides a neuroscience-oriented framework for efficiently setting up and running simulations. CoBeL-RL offers a set of virtual environments, e.g., T-maze and Morris water maze, which can be simulated at different levels of abstraction, e.g., a simple gridworld or a 3D environment with complex visual stimuli, and set up using intuitive GUI tools. A range of RL algorithms, e.g., Dyna-Q and deep Q-network algorithms, is provided and can be easily extended. CoBeL-RL provides tools for monitoring and analyzing behavior and unit activity, and allows for fine-grained control of the simulation via interfaces to relevant points in its closed-loop. In summary, CoBeL-RL fills an important gap in the software toolbox of computational neuroscience.
(From Diekmann et al. (2023), **https://doi.org/10.3389/fninf.2023.1134405**)

-----------------------------------------------------------
**Installation Guide**

1. Download or clone the project using git. You can do this by typing the following into your command line : 

`git clone https://gitlab.ruhr-uni-bochum.de/cns/1-frameworks/CoBeL-RL.git` 

See [Gitlab documentation](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#clone-a-repository) for more details and options for cloning repositories.

2. Install CoBeL-RL through [pip](https://pypi.org/project/pip/). See install options below for installing optional packages.
Type the following into your command line : 

`pip3 install /path/to/cloned/project/`


3. These steps are sufficient to install CoBeL-RL. Get started by running one of the demos below. For new users, we recommend the Dyna-Q demo.

<details></summary>
<summary>

Install Options
</summary>

* If you intend to use Keras-RL agents install with the option:

`[keras-rl]`

* If you intend to use CoBeL-RL's Deep RL agents install with the option:

`[tensorflow]`

`[torch]`

* If you intend to use Unity environments install with the option:

`[unity]`

</details>

--------------------------------------------------------------------------------

**Run the Dyna-Q Demo**

1. Go to demo folder. From the command line, 

`cd ~/demo/gridworld/`

2.  Start the demo simulation by typing

`python gridworld_dyna_q_demo.py`

This should open the CoBeL-RL visualizer and run the demo.

-----------------------------------------------------------------------------------

**Run the DQN Demo**

1.  Install PyTorch either as an optional requirement or directly from the official website:

`https://pytorch.org/`

2.  Choose one of the 3D simulators below and follow the listed instructions:

<details></summary>
<summary>
Blender
</summary>


1. Download and install [Blender2.79b](https://download.blender.org/release/Blender2.79/)

You will need to set up an environment variable for the Blender path to run the demos.

`export BLENDER_EXECUTABLE_PATH="/path/to/blender2.79b/"`

Alternatively, you can set the variable manually in Python before running the script.

`import os`

`os.environ['BLENDER_EXECUTABLE_PATH'] = '/path/to/blender2.79b/'`

Note : If you plan to use the framework more than once, it is useful to set this variable permanently. On Linux distributions, you can do this by editing the .bashrc file. 

2. Go to demo folder

`cd ~/demo/simple_grid_graph_demo/`

3. Start the demo simulation:

`python simple_grid_graph_demo.py`

**Note: The Blender path can also be passed as a parameter to the Blender frontend module. For your own simulations it is therefore not necessary to set a permanent variable.**

</details>


<details>

<summary>
Unity
</summary>


You have two options to run a demo.

1. You can use the precompiled versions of the unity environments.
    
    You have to set a system variable named 'UNITY_ENVIRONMENT_EXECUTABLE' to the path of the downloaded environments
    
    > export UNITY_ENVIRONMENT_EXECUTABLE=PATH_TO_ENV_EXEC
    
    and run the unity_demo.py in the demo/unity_demo folder
    
    > python3 demo/unity_demo/unity_demo.py
        
2. To build the environments yourself and custom environments do the following:

    * you need to download and install the 'Unity Hub': **https://docs.unity3d.com/Manual/GettingStartedInstallingHub.html**
    
    * the adapted version of mlagents: **https://ruhr-uni-bochum.sciebo.de/s/8GUszMEC7LgzS7V**
    
    * and the unitypackage for the environments: **https://ruhr-uni-bochum.sciebo.de/s/gdphysRY1P7pAyT**
    
    * the unitypackage is also available in the git folder: **environments/environments_unity/source/unity_environments.unitypackage**
    
    Then you set up a new project with unity. See: **https://docs.unity3d.com/560/Documentation/Manual/GettingStarted.html**
    
    To import the 'mlagents' framework into your project you select the 'Window/PackageManager' menu item in the editor, 
    then choose 'Add Package From Disk' in the top left corner and open the 'package.json' in the 'ml-agents/com.unity.ml-agents' folder.
    
    To import the environments you select the menu item 'Assets/Import Package/Custom Package' in the editor and open the 
    'unity_resources.unitypackage' you downloaded.

3. The other option is to connect the interface directly with the Unity editor.
    
    You start training an environment by opening a scene in the 'Assets/Scenes' folder with the 'Project Explorer' of the editor, 
    running the unity_demo.py first and pressing the 'Play' button at the top of the editor screen.
    
    **Note: the demo tries to do option 1) automatically when 'UNITY_ENVIRONMENT_EXECUTABLE' variable is set.**
    
 </details> 

 
 <details>

<summary>
Godot
</summary>


1. Download the Godot build (build contains executables for Linux and Windows).

`https://ruhr-uni-bochum.sciebo.de/s/dSGaGtflsNTqYRW`

2. You will need to set up an environment variable for the Godot path to run the demos.

`export GODOT_EXECUTABLE_PATH="/path/to/godot/"`

3. Alternatively, you can set the variable manually in Python before running the script.

`import os`

`os.environ['GODOT_EXECUTABLE_PATH'] = '/path/to/godot/'`
    
4. Go to demo folder.

`cd ~/demo/godot/`

5. Start the demo simulation:

`python godot_demo_grid_graph.py`

**Note: The Godot path can also be passed as a parameter to the Godot frontend module. For your own simulations it is therefore not necessary to set a permanent variable.**
  
 </details>
 
-------------------------------------------------------------------------------------------

**List of publications**

* Zeng X, Diekmann N, Wiskott L and Cheng S (2023) Modeling the function of episodic memory in spatial learning. Front. Psychol. 14:1160648. [[DOI]]( https://doi.org/10.3389/fpsyg.2023.1160648)
* Diekmann, N., Vijayabaskaran, S., Zeng, X., Kappel, D., Menezes, M. C., & Cheng, S. (2023). CoBeL-RL: A neuroscience-oriented simulation framework for complex behavior and learning. In Frontiers in Neuroinformatics (Vol. 17). Frontiers Media SA. [[DOI]](https://doi.org/10.3389/fninf.2023.1134405)
* Kappel, D., & Cheng, S. (2023). Global remapping emerges as the mechanism for renewal of context-dependent behavior in a reinforcement learning model. BioRXiV. [[DOI]](https://doi.org/10.1101/2023.10.27.564433) 
* Diekmann, N., & Cheng, S. (2023). A model of hippocampal replay driven by experience and environmental structure facilitates spatial learning. In eLife (Vol. 12). eLife Sciences Publications, Ltd. [[DOI]](https://doi.org/10.7554/elife.82301) 
* Vijayabaskaran, S., & Cheng, S. (2022). Navigation task and action space drive the emergence of egocentric and allocentric spatial representations. In D. Bush (Ed.), PLOS Computational Biology (Vol. 18, Issue 10, p. e1010320). Public Library of Science (PLoS). [[DOI]](https://doi.org/10.1371/journal.pcbi.1010320) 
* Walther, T., Diekmann, N., Vijayabaskaran, S. et al. Context-dependent extinction learning emerging from raw sensory inputs: a reinforcement learning approach. Sci Rep 11, 2713 (2021). [[DOI]](https://doi.org/10.1038/s41598-021-81157-z) 

--------------------------------------------------------------------------------------------
**Code Contributors**

Present Contributors: Diekmann, N., Vijayabaskaran, S., Zeng. X

Past Contributors : Chaves Menezes, M., Kappel, D., Panagiotou, F., Walther, T.

Students : Jungeilges. A, Prasad. S
