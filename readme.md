## **CoBeL-RL** ##
#### Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks ####

This is a new version of CoBeL-RL.
If you require an older version, download the appropriately tagged commit/branch.

----------------------------

**Installation Guide**

`pip3 install -r requirements.txt`

* Clone the project.

`git clone https://gitlab.ruhr-uni-bochum.de/cns/1-frameworks/CoBeL-RL.git`  

*  Then, install CoBeL-RL through pip.

`pip3 install /path/to/cloned/project/`

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

------------------------------

**Run the Dyna-Q Demo**

* Go to demo folder

`cd ~/demo/gridworld/`

*  Start the demo simulation:

`python gridworld_dyna_q_demo.py`

------------------------------

**Run the DQN Demo**

*  Install PyTorch either as an optional requirement or directly from the official website:

`https://pytorch.org/`

*  Choose one of the 3D simulators below and follow the listed instructions:

<details></summary>
<summary>
Blender
</summary>


* Download and install Blender2.79b from:  

`https://download.blender.org/release/Blender2.79/`

* You will need to set up an environment variable for the Blender path to run the demos.

`export BLENDER_EXECUTABLE_PATH="/path/to/blender2.79b/"`

* Alternatively, you can set the variable manually in Python before running the script.

`import os`

`os.environ['BLENDER_EXECUTABLE_PATH'] = '/path/to/blender2.79b/'`

If you plan to use the framework more than once, it is useful to set this variable permanently. On Linux distributions, you can do this by editing the .bashrc file. 

* Go to demo folder

`cd ~/demo/simple_grid_graph_demo/`

*  Start the demo simulation:

`python simple_grid_graph_demo.py`

**Note: The Blender path can also be passed as a parameter to the Blender frontend module. For your own simulations it is therefore not necessary to set a permanent variable.**

</details>


<details>

<summary>
Unity
</summary>


You got two options to run a demo.

*  You can use the precompiled versions of the unity environments.
    
    You have to set a system variable named 'UNITY_ENVIRONMENT_EXECUTABLE' to the path of the downloaded environments
    
    > export UNITY_ENVIRONMENT_EXECUTABLE=PATH_TO_ENV_EXEC
    
    and run the unity_demo.py in the demo/unity_demo folder
    
    > python3 demo/unity_demo/unity_demo.py
        
* To build the environments yourself and custom environments do the following:

    * you need to download and install the 'Unity Hub': **https://docs.unity3d.com/Manual/GettingStartedInstallingHub.html**
    
    * the adapted version of mlagents: **https://ruhr-uni-bochum.sciebo.de/s/8GUszMEC7LgzS7V**
    
    * and the unitypackage for the environments: **https://ruhr-uni-bochum.sciebo.de/s/gdphysRY1P7pAyT**
    
    * the unitypackage is also available in the git folder: **environments/environments_unity/source/unity_environments.unitypackage**
    
    Then you set up a new project with unity. See: **https://docs.unity3d.com/560/Documentation/Manual/GettingStarted.html**
    
    To import the 'mlagents' framework into your project you select the 'Window/PackageManager' menu item in the editor, 
    then choose 'Add Package From Disk' in the top left corner and open the 'package.json' in the 'ml-agents/com.unity.ml-agents' folder.
    
    To import the environments you select the menu item 'Assets/Import Package/Custom Package' in the editor and open the 
    'unity_resources.unitypackage' you downloaded.

* The other option is to connect the interface directly with the Unity editor.
    
    You start training an environment by opening a scene in the 'Assets/Scenes' folder with the 'Project Explorer' of the editor, 
    running the unity_demo.py first and pressing the 'Play' button at the top of the editor screen.
    
    **Note: the demo tries to do option 1) automatically when 'UNITY_ENVIRONMENT_EXECUTABLE' variable is set.**
    
 </details> 

 
 <details>

<summary>
Godot
</summary>


*  Download the Godot build (build contains executables for Linux and Windows).

`https://ruhr-uni-bochum.sciebo.de/s/dSGaGtflsNTqYRW`

* You will need to set up an environment variable for the Godot path to run the demos.

`export GODOT_EXECUTABLE_PATH="/path/to/godot/"`

* Alternatively, you can set the variable manually in Python before running the script.

`import os`

`os.environ['GODOT_EXECUTABLE_PATH'] = '/path/to/godot/'`
    
* Go to demo folder.

`cd ~/demo/godot/`

* Start the demo simulation:

`python godot_demo_grid_graph.py`

**Note: The Godot path can also be passed as a parameter to the Godot frontend module. For your own simulations it is therefore not necessary to set a permanent variable.**
  
 </details>    
