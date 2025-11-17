## Instructions for automated training

### The current script supports all trainings, but the setup is very specific. To setup the environment, you will need to:
1. [Create the executables you want to use](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html) with the SAME NAME as the 2nd line of the YAML file (bahaviour name). That would be 3DBall.app for 3Dball, etc

### To run the script, simply use:

```bash
python3 training_script.py <executable-folder>
```
where executable-folder is the folder containing the executables generated in unity. The folder sturcture should look like this:


```bash
data-collection/
├── executables/
│   ├── 3DBall.app
│   ├── GridFoodCollector.app
│   └── ...
├── training_script.py
```
