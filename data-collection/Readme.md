## Instructions for automated training

### The current script supports all trainings, but the setup is very specific. I will work on making it more automated. To setup the environment, you will need to:
1. [Create the executables you want to use](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html) with the SAME NAME as the 2nd line of the YAML file (bahaviour name). That would be 3DBall.app for 3Dball, etc
2. Create a folder with multiple yaml files

### To run the script, simply use:

```bash
python3 training_script.py <YAML-folder> <executable-folder>
```
where YAML-folder is a folder containing multiple YAML files and executable-folder is the folder containing the executables generated in unity. YAML file generation is not yet complete, we are still waiting for [issue #69](https://github.com/Ninjagoat73/project2-1_group15/issues/69) to be closed so we can actually start training
 The folder sturcture should look like this:

```bash
data-collection/
├── executables/
│   ├── 3DBall.app
│   ├── GridFoodCollector.app
│   └── ...
├── yamls/
│   └── 3DBall.yaml
│   ├── FoodCollector.yaml (note that the bahaviour name is actually GridFoodCollector)
│   └── ...

```
