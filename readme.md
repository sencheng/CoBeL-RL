This is the "Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks" **(CoBeL-RL)**.

Version by date:

2020/03/19: **very preliminary**, initial version, this version might not be stable, and might have other issues. 
Also, the structure of the project might undergo changes. All in all, this version is meant as an initial demonstration 
of the project. If you try it, please **keep this in mind**. If you find issues, and/or have ideas for enhancing the 
system structure, etc., please enter them in the issue tracker.

**Getting started**:

If you want to quickly try the simulator:

* Install Blender. If Blender is not already present in your system, download and unpack Blender 2.79b from 
https://download.blender.org/release/Blender2.79/
    * Go to the place where your download is, e.g. 
    > cd ~/Downloads
    * Extract the files into a folder of your choice, e.g. if you downloaded blender-2.79b-linux-glibc219-x86_64.tar.bz2 
     and you want to install in /etc/opt:
    > bzip2 -vfd blender-2.79b-linux-glibc219-x86_64.tar.bz2
                                                                                                                                                                                                                                                                                                                                                                                             
    > sudo tar xvf blender-2.79b-linux-glibc219-x86_64.tar -C /etc/opt
                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                          


* Clone the project. (This installs to your home folder, so ~/CoBel-RL)
    > git clone https://gitlab.ruhr-uni-bochum.de/cns/1-frameworks/CoBeL-RL.git

* Setup Python. Create a virtual environment by typing the following commands in the 
terminal: (you can change the path name to whatever you want)
    > python3 -m venv ~/cobel_venv

    > virtualenv cobel_venv

    * Activate your virtual environment with 
    > source ~/cobel_venv/bin/activate

    * Install the dependencies in requirements.txt e.g. by running the following
 command from the CoBel-RL directory:
    >pip install -r requirements.txt --no-index

    Alternatively if you are working from the institute you can activate the provided virtual environment with 
    > source /groups/cns/venv/cobel_rl/venv/bin/activate


* Setup the environment variables. 
    * Make sure that your 'PYTHONPATH' environment variable includes the project's root directory.

        With the virtual environment activated, navigate to your project folder
        > cd ~/CoBeL-RL

        Add the project's directory to the PYTHONPATH enviroment variable
        >export PYTHONPATH="$PWD"

    * Set a 'BLENDER_EXECUTABLE_PATH' environment variable that points to the path containing the 'Blender' executable, e.g.

        > export BLENDER_EXECUTABLE_PATH='/etc/opt/blender-2.79b-linux-glibc219-x86_64/'

    To make it **permanent** you have to edit the ~/.bash_profile file and add both variables  there
    (The $PWD command only gives the current directory, here you would have set it to ~/CoBeL-RL yourself)

* Go to the demo folder
> cd ~/CoBeL-RL/demo/simpleGridGraphDemo/

* Start the demo project: 
>python3 simpleGridGraphDemo.py

* If you want to work with a GUI editor you will need to add the environment variable manually.
In Pycharm this can be done by adding them to the Run configuration
(Run -> Run... -> Edit Configurations... -> Environment variables)