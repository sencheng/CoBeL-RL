This is the "Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks" **(CoBeL-RL)**.

Version by date:

2020/03/19: **very preliminary**, initial version, this version might not be stable, and might have other issues. Also, the structure of the project might undergo changes. All in all, this version is meant as an initial demonstration of the project. If you try it, please **keep this in mind**. If you find issues, and/or have ideas for enhancing the system structure, etc., please enter them in the issue tracker.

**Getting started**:

If you want to quickly try the simulator:
* clone the project to some folder 'mySimulatorFolder' (might be your home folder)

* Create a virtual environment by typing the following commands in the 
terminal: (you can change the path name to whatever you want)
> python3 -m venv /cobel_venv

> virtualenv cobel_venv

* Activate your virtual environment with 
> source /cobel_venv/bin/activate

* Install the dependencies in requirements.txt e.g. by running the following
 command from the CoBel-RL directory:
>pip install -r requirements.txt --no-index

Alternatively if you are working from the institute activate the provided virtual environment with 
> source /groups/cns/venv/cobel_rl/venv/bin/activate

* make sure that your 'PYTHONPATH' environment variable includes **'mySimulatorFolder/CoBel-RL'**.
* go to **.../mySimulatorFolder/CoBel-RL/demo/simplGridGraphDemo/**
* start the demo project: **python3 simpleGridGraphDemo.py**
