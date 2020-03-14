# Target-driven Visual Navigation Model using Deep Reinforcement Learning
[Docker Image](https://hub.docker.com/repository/docker/denmonz/ai2thor)<br>
[Dockerfile](Dockerfile)

## Problem Statement

Indoor visual navigation has many challenges. Where do you get the training data? How can you guarantee that your agent will generalize well? How can you ensure your agent will adapt well to dynamic environments? Our project is built off of the deep siamese actor-critic model introduced in the following paper:  <br><br>**[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](https://arxiv.org/abs/1609.05143)**
<br>
[Yuke Zhu](http://web.stanford.edu/~yukez/), Roozbeh Mottaghi, Eric Kolve, Joseph J. Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi
<br>
[ICRA 2017, Singapore](http://www.icra2017.org/)
<br><br>
The agent aims for environmental, goal/target, and real-world generalizability. The original agent utilized the first version of AI2Thor, and we transitioned the agent into the latest version of [AI2Thor](https://ai2thor.allenai.org/).

<p align="center">
  <img src="/images/model.png" data-canonical-src="/images/networkArchitecture.png" width="450"/><br>
  <em>Image retrieved from Zhu et. al., 2017.</em>
</p>

## Input and Output

The model takes as input an observation and goal image, both of size (224x224x3), and the output is one of the four following actions to be taken by the agent at any time step: "Move Ahead", "Rotate Right", "Rotate Left", "Move Back".
We extract feature maps of the input images via the second to last layer of ResNet 50 (by utilizing [Keras](https://keras.io/applications/#resnet)). These features are dimensionally reduced representations of the original images, and are used for the deep learning component of our model.

<p align="center">
  <img src="/images/networkArchitecture.png" data-canonical-src="/images/networkArchitecture.png" width="450"/><br>
  <em>Image retrieved from Zhu et. al., 2017.</em>
</p>

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

## Our Amendments
* Upgrades the agent's original AI2Thor environment to the latest AI2Thor environment.
* Implemented online feature extraction for agent evaluation.
* Updated keyboard_agent.py to navigate the new environment and capture goal images.

## Future Work
* Cache ResNet50 features for quicker training and evaluation.
* Replace ResNet for RedNet and utilize depth as input to the model.

## License
MIT
