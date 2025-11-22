# Project 2-1: AI and ML with Unity

- Group number: 15
- BSc Computer Science
- Academic Year: 2025/2026
- Maastricht University

## Description

This repository is a fork from [ml-agents](https://github.com/DennisSoemers/ml-agents). Our focus during this project will be to explore the ML Agents framework and collect data from RL runs on the Unity engine. This data will then be used to train ML models to predict certain properties of deep RL runs.

## Prerequisites:

- Conda
- Python 3.10.12
- Unity 6000.0 or later

## Installation instructions:
Since we've modified the ML Agents library, we need to compile it ourselves (not download it from pip). These are the instructions: 

1. Clone repository from Github
2. cd into the repository
3. Create a conda environment using ```conda create -n mlagents python=3.10.12 && conda activate mlagents```
4. Build grpcio wheel ``` conda install "grpcio=1.48.2" -c conda-forge ```
5. Install ml-agents-envs ``` cd ml-agents-envs && pip install -e . ```
6. Install ml-agents ``` cd ../ml-agents && pip install -e . ```

Use [this guide](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html) for troubleshooting

## Project status
The group is currently experimenting with the ML Agents framework and Unity. We are also working on coming up with the research question and thinking of ways to collect data for phase 2 and 3

## Future work

## Limitations
