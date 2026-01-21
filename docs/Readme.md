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
5. Install our requirements ``` pip install -r requirements.txt ``` (or pip3)
6. Install ml-agents-envs ``` cd ml-agents-envs && pip install -e . ```
7. Install ml-agents ``` cd ../ml-agents && pip install -e . ```
8. Unzip data.zip in ``` data-collection/data/data.zip``` (overwrite the files if prompted)

Use [this guide](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html) for troubleshooting

## Generating data
In order to train Unity models and generate data, please refer to [this readme](https://github.com/Ninjagoat73/project2-1_group15/tree/develop/data-collection#readme)

## Running the Machine Learning experiments
Go to [this folder](https://github.com/Ninjagoat73/project2-1_group15/tree/develop/data-analysis). There are 5 different folders, each one corresponding to one research question. Each folder has a readme file with instructions on how to reproduce each experiment

## Project status
The group has finished coding the pipeline for data generation and answered [five research questions](https://github.com/Ninjagoat73/project2-1_group15/tree/develop/data-analysis)
