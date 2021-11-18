![Screenshot](benchmark.png)

--------------------------
## **CoBeL-RL** is the "Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks" .

** Getting started: **

<details>
<summary>
**Installation:**
</summary>
    
* Blender 2.79b
     `https://download.blender.org/release/Blender2.79/`
* Set up virtual environment and install requirements.txt
</details>

<details>
<summary>
** Run Demo: **
</summary>

*  Clone the project
>   `git clone https://gitlab.ruhr-uni-bochum.de/cns/1-frameworks/CoBeL-RL.git`

* Activate virtual environment
* Go to demo folder
>   `cd ~/CoBeL-RL/demo/simpleGridGraphDemo/`
*  Start the demo project:
>   `python3 simpleGridGraphDemo.py`

</details>


<details>
<summary>
** Setup the environment variables: **
</summary>

*  Set a 'BLENDER_EXECUTABLE_PATH' environment variable that points to the path containing the 'Blender' executable, e.g:
>   `export BLENDER_EXECUTABLE_PATH='/etc/opt/blender-2.79b-linux-glibc219-x86_64/'`
* Make sure that your 'PYTHONPATH' environment variable includes the project's root directory. 
  
    - With the virtual environment activated, navigate to your project folder
       > `cd ~/CoBeL-RL`
    
    - Add the project's directory to the PYTHONPATH enviroment variable
       > `export PYTHONPATH="$PWD"`

</details>


>    To make it **permanent** you have to edit the ~/.bash_profile file and add both variables  there
>    (The $PWD command only gives the current directory, here you would have set it to ~/CoBeL-RL yourself)



__Preliminary information, needs further checking: It seems there are some issues with 'tensorflow', version 1.5.0 and the employed 'python' version. If you experience such compatibility problems and have a 'python' version >=3.7, it might help to downgrade 'python' to a version >=2.7 and <=3.6. However, this is just a preliminary hint, the issue will have to be further explored.__



________________________________________________________________________________________________

**Unity Interface**

If you want to try the unity interface demo, you need to perform the the steps described above first.

Please report all bugs you find :)

**Password for all 'Sciebo' downloads: cobel_rl**

You got two options to run a demo.

*  You can use the precompiled versions of the unity environments.

    * Linux: **https://ruhr-uni-bochum.sciebo.de/s/3iFYDgzGxLJ57tv**
    
    * Windows **https://ruhr-uni-bochum.sciebo.de/s/F56wugRAdWRfTj3**
    
    You have to set a system variable named 'UNITY_ENVIRONMENT_EXECUTABLE' to the path of the downloaded environments
    
    > export UNITY_ENVIRONMENT_EXECUTABLE=PATH_TO_ENV_EXEC
    
    and run the unity_demo.py in the demo/unity_demo folder
    
    > python3 demo/unity_demo/unity_demo.py
        
* The other option is to install the unity editor and connect the interface directly with the editor.

    * you need to download and install the 'Unity Hub': **https://docs.unity3d.com/Manual/GettingStartedInstallingHub.html**
    
    * the adapted version of mlagents: **https://ruhr-uni-bochum.sciebo.de/s/8GUszMEC7LgzS7V**
    
    * and the unitypackage for the environments: **https://ruhr-uni-bochum.sciebo.de/s/gdphysRY1P7pAyT**
    
    * the unitypackage is also available in the git folder: **environments/environments_unity/source/unity_environments.unitypackage**
    
    Then you set up a new project with unity. See: **https://docs.unity3d.com/560/Documentation/Manual/GettingStarted.html**
    
    To import the 'mlagents' framework into your project you select the 'Window/PackageManager' menu item in the editor, 
    then choose 'Add Package From Disk' in the top left corner and open the 'package.json' in the 'ml-agents/com.unity.ml-agents' folder.
    
    To import the environments you select the menu item 'Assets/Import Package/Custom Package' in the editor and open the 
    'unity_resources.unitypackage' you downloaded.
    
    You start training an environment by opening a scene in the 'Assets/Scenes' folder with the 'Project Explorer' of the editor, 
    running the unity_demo.py first and pressing the 'Play' button at the top of the editor screen.
    
    **Note: the demo tries to do option 1) automatically when 'UNITY_ENVIRONMENT_EXECUTABLE' variable is set.**
    
     
     
