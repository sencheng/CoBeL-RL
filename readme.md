## **CoBeL-RL** ##
#### Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks ####

This is a new version of CoBeL-RL. For the older version, please use the tagged commit v1.0.
---------------------------
**Getting started:**  

To quickly get started and test the system, you can run one of the demos. 

----------------------------

**Install requirements**

    
* Many simulations depend on Blender2.79b. If you wish to use the Unity editor, skip this step. Download and install Blender2.79b at:  

     `https://download.blender.org/release/Blender2.79/`  

Note : Only v2.79b is supported. Newer versions of Blender will not work with the system.  
* Install required python packages  

`pip install -r requirements.txt`

------------------------------

<details></summary>
<summary>
Run Demo
</summary>


*  First, clone the project.

`git clone https://gitlab.ruhr-uni-bochum.de/cns/1-frameworks/CoBeL-RL.git`  

*  Then, move the cloned project to the `site-packages` of your Python installation and rename the project directory to `cobel`.

* You will need to set up an environment variable for the Blender path to run the demos.

`export BLENDER_EXECUTABLE_PATH="/path/to/blender2.79b/"`

* Alternatively, you can set the variable manually in Python before running the script.

`import os`

`os.environ['BLENDER_EXECUTABLE_PATH'] = '/path/to/blender2.79b/'`

If you plan to use the framework more than once, it is useful to set this variable permanently. On Linux distributions, you can do this by editing the .bashrc file. 

* Go to demo folder
`cd ~/cobel/demo/simple_grid_graph_demo/`

*  Start the demo simulation:
`python simple_grid_graph_demo.py`

**Note: The Blender path can also be passed as a parameter to the Blender frontend module. For your own simulations it is therefore not necessary to set a permanent variable.**

</details>

------------------------------

<details>
________________________________________________________________________________________________
<summary>
Unity interface (Skip if you plan to work with Blender)
</summary>

If you want to try the unity interface demo, you need to perform the the steps described above first.

Please report all bugs you find :)

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
 
 ------------------------------
 
 <details>
________________________________________________________________________________________________
<summary>
Godot interface (TBD)
</summary>

Expected to be available after summer semester 2022.
  
 </details>    
