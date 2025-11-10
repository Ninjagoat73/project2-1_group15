## Instructions for automated training

### The current script is very limited and only supports 3DBall training. I will work on making different types of agents compatible. To setup the environment, you will need to:
1. [Create the 3DBall executable](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html)
2. Create a folder with multiple 3DBall yaml files

### To run the script, simply use:

```bash
python3 training_script.py <YAML-folder> <3DBall-executable>
```
where YAML-folder is a folder containing multiple YAML files (for 3DBall currently) and 3DBall-executable is the executabl generated in unity. YAML file generation is not yet complete, we are still waiting for [issue #69](https://github.com/Ninjagoat73/project2-1_group15/issues/69) to be closed so we can actually start training
