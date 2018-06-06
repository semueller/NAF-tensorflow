# Normalized Advantage Functions (NAF) in TensorFlow

TensorFlow implementation of [Continuous Deep q-Learning with Model-based Acceleration](http://arxiv.org/abs/1603.00748).

![algorithm](https://github.com/carpedm20/naf-tensorflow/blob/master/assets/algorithm.png)

## Environments:

- InvertedPendulum-v1
- InvertedDoublePendulum-v1
- Reacher-v1
- HalfCheetah-v1
- Swimmer-v1
- Hopper-v1
- Walker2d-v1
- Ant-v1
- HumanoidStandup-v1


## Installation and Usage

The code depends on outdated software, until it is updated to work with current versions of gym/ tensorflow /mujoco, set up a custom virtualenv (eg with conda) for this and run setup.sh:

    $ conda create --name naf python=2.7
    $ source actiavate naf
    $ ./setup.sh

To train a model for an environment with a continuous action space:

    $ python main.py --env=InvertedPendulum-v1 --is_train=True
    $ python main.py --env=InvertedPendulum-v1 --is_train=True --display=True

To test and record the screens with gym:

    $ python main.py --env=InvertedPendulum-v1 --is_train=False
    $ python main.py --env=InvertedPendulum-v1 --is_train=False --monitor=True

To visualize training results with tensorboard:

    # activate some other python environment with python=3.5
    $ ./tensorboard --logdir=./logs

The TensorBoard binary provided here is TensorBoard 0.4.0rc3 and the only one I was able to actually visualize the logs with







## Results


## References

- [rllab](https://github.com/rllab/rllab.git)

## Original Author 

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
Original [git](https://github.com/carpedm20/NAF-tensorflow)
