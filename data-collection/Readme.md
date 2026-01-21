## Instructions for automated training

### The current script supports most unity agents, but the setup is very specific. To setup the environment, you will need to:
1. [Create the executables you want to use](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Learning-Environment-Executable.html) with the SAME NAME as the 2nd line of the YAML file (bahaviour name). That would be 3DBall.app for 3Dball, etc
2. Have the 3 csv files inside folder "data"
3. Have a folder called "results"
```bash
mkdir results
```
3. [This YAML file](https://github.com/Ninjagoat73/project2-1_group15/blob/develop/data-collection/test_batch_config.yaml) enables the user to configure what games will be trained and how many YAML files for each game will be generated. Make sure to only set the game values to true if you have the appropriate executable for that game, as mentioned in step 1

### To run the script, simply use:

```bash
python3 training_scipt.py <path-to-executable-folder> <path-to-results-folder> <path-to-data-folder> <path-to-batch-yaml-file>
```
where executable-folder is the folder containing the executables generated in unity. The folder structure should look like this: (it will look different for Windows users, you need to name the folder 3DBall, etc)


```bash
data-collection/
├── executables/
│   ├── 3DBall.app
│   ├── GridFoodCollector.app
│   └── ...
├── results/
├── data/
│   ├── static.csv
│   ├── training.csv
│   └── static.csv
├── training_script.py
```
