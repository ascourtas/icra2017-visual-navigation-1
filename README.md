## Target-driven Visual Navigation Model using Deep Reinforcement Learning

![THOR scene samples](http://web.stanford.edu/~yukez/images/img/thor_examples.png "THOR scene samples")

## Introduction

This repocitory provides a Tensorflow implementation of the deep siamese actor-critic model for indoor scene navigation introduced in the following paper:

**[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](http://web.stanford.edu/~yukez/papers/icra2017.pdf)**
<br>
[Yuke Zhu](http://web.stanford.edu/~yukez/), Roozbeh Mottaghi, Eric Kolve, Joseph J. Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi
<br>
[ICRA 2017, Singapore](http://www.icra2017.org/)

## Setup
The code utilizes [Tensorflow API r2.0](https://www.tensorflow.org/api_docs/), but is backwards compatible with all Tensorflow 1.0 methods from the original repository. The project was built and tested on macOS Mojave (10.14.6) using Python 3.7.6, but running the project via Docker should eliminate any OS-dependent issues. For the full list of dependencies, see the `requirements.txt` file at the root of this repo. 

### Getting h5 Files
This repo utilizes a pretrained A3C model, with all important model object information encoded in [hdf5](http://www.h5py.org/) dumps. To download the h5 dumps, first ensure that you have the `wget` utility installed -- here are installation instructions for [Linux distros](https://www.tecmint.com/install-wget-in-linux/) and [macOS (Homebrew install recommended)](https://www.fossmint.com/install-and-use-wget-on-mac/). 

Then, run the following from the root of the repository:
```
bash
./data/download_scene_dumps.sh
```
The h5 files should be located in the `/data` folder.

If `wget` does not install properly, you can always download the files by clicking [here](http://vision.stanford.edu/yukezhu/thor_v1_scene_dumps.zip), unzipping the download, and moving the files to the `/data` folder. You should then rerun the above bash script to rename the files appropriately. 

### Using Docker 
1. Ensure you have Docker installed. See the [Docker installation instructions](https://docs.docker.com/install/) for more details; you would use the Desktop version for macOS or Windows, and the Server version for Linux distros. Check that Docker installed properly by running the following in your terminal:
```
docker --version
```
2. From the root of the repository, run the following to build the Docker image (note this may take a while):
```
docker build -t ai2thor:1.0.0 .
```

## Running the model
We utilize a pre-trained model from the original repository, therefore training is unnecessary. The following instructions are for running the evaluation of said model. The output of the stats on each episode will be displayed in the terminal.

Run the Docker container (note that it may take >30 min before evaluation stats are printed):
```
docker run ai2thor:1.0.0
```

## Other Utilities

We've provided a modified `keyboard_agent.py` script that allows you to load a scene dump and use the arrow keys to navigate a scene. The script also lets you take screenshots to use as target images for your own evaluations, or capture depth images if you wish to experiment with those. 

The commands are:
> Use WASD keys to move the agent.<br>
> Use QE keys to move the camera.<br>
> Press I to switch between RGB and Depth views.<br>
> Press P to save an image of the current view.<br>
> Press R to reset agent's location.<br>
> Press F to quit.

We have not created a Docker environment to run this, but feel free to explore this utility by installing the dependencies on your local machine. To run the script, [install Python 3.7.6](https://www.python.org/downloads/), [`pip`](https://pip.pypa.io/en/stable/installing/), and download the dependencies outlined in `requirements.txt` by running `pip install -r requirements.txt`. Then, install [Unity via Unity Hub](https://unity3d.com/get-unity/download). 

Start the utility via:
```
python keyboard_agent.py 
```
This should open an AI2-THOR window for you to navigate within. 

## Training and Evaluation
The parameters for training and evaluation are defined in ```constants.py```. The most important parameter is ```TASK_LIST```, which is a dictionary that defines the scenes and targets to be trained and evaluated on. The keys of the dictionary are scene names, and the values are a list of location ids in the scene dumps, i.e., navigation targets. We use a type of asynchronous advantage actor-critic model, similar to [A3C](https://arxiv.org/abs/1602.01783), where each thread trains for one target of one scene. Therefore, make sure the number of training threads ```PARALLEL_SIZE``` is *at least* the same as the total number of targets. You can use more threads to further parallelize training. For instance, when using 8 threads to train 4 targets, 2 threads will be allocated to train each target.

The model checkpoints are stored to ```CHECKPOINT_DIR```, and Tensorboard logs are written in ```LOG_FILE```. To train a target-driven navigation model, run the following script:
```bash
# train a model for targets defined in TASK_LIST
python train.py
```

For evaluation, we run 100 episodes for each target and report the mean/stddev length of the navigation trajectories. To evaluate a model checkpoint in ```CHECKPOINT_DIR```, run the following script:
```bash
# evaluate a checkpoint on targets defined in TASK_LIST
python evaluate.py
```

## Acknowledgements
I would like to acknowledge the following references that have offered great help for me to implement the model.
* ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
* [David Silver's Deep RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [muupan's async-rl repo](https://github.com/muupan/async-rl/wiki)
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)

## Citation
Please cite our ICRA'17 paper if you find this code useful for your research.
```
@InProceedings{zhu2017icra,
  title = {{Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning}},
  author = {Yuke Zhu and Roozbeh Mottaghi and Eric Kolve and Joseph J. Lim and Abhinav Gupta and Li Fei-Fei and Ali Farhadi},
  booktitle = {{IEEE International Conference on Robotics and Automation}},
  year = 2017,
}
```

## License
MIT
