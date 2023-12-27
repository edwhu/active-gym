from setuptools import find_packages, setup

setup(
    name='active_gym',
    version='0.2',
    author='Jinghuan Shang',
    description=('Gym-like wrapper to implement Active Reinforcement Learning environments based on existing environments'),
    keywords='active-reinforcement-learning active-rl reinforcement learning rl gym atari deepmind robosuite',
    packages=find_packages(),
    install_requires=[
        "mujoco",
        "mujoco-py>=2.1.2.14",
        "atari-py",
        "dm-control==1.0.11",
        "gym",
        "torch",
        "torchvision",
        "transform3d"
    ],
    extras_require={
        "robosuite": [
            "robosuite",
        ],
        "rlbench": [
            "git+https://github.com/stepjam/PyRep.git",
            "git+https://github.com/stepjam/RLBench.git"
        ]
    },
)