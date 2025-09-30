import pandas as pd
import psutil
import platform
from datetime import datetime
from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name="D:\project2-1\Builds/UnityEnvironment.exe")

env.reset()

behavior_name = list(env.behavior_specs.keys())[0]

print("Behavior:", behavior_name)

spec = env.behavior_specs[behavior_name]
print("Observation shape", spec.observation_specs)
print("Action shape", spec.action_spec)

data = []

for episode in range(10):
    env.reset()
    step_count = 0
    done = False

    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        action = spec.action_spec.random_action(len(decision_steps))
        continuous_actions = action.continuous
        discrete_actions = action.discrete
        env.set_actions(behavior_name, action)
        env.step()
        i=0

        # Loop through agents that need a decision
        for agent_id, agent_step in decision_steps.items():

            obs = agent_step.obs[0]  # vector observation
            reward = agent_step.reward
            cont_action = continuous_actions[i] if continuous_actions is not None else None
            discrete_action = discrete_actions[i] if discrete_actions is not None else None


            i += 1
            data.append({
                "episode": episode,
                "step": step_count,
                "agent_id": agent_id,
                "observation": obs.tolist(),
                "continuous_action": cont_action.tolist() if cont_action is not None else None,
                "discrete_action": discrete_action.tolist() if discrete_actions is not None else None,
                "reward": reward,
                "done": False
            })

        # Loop through agents whose episode ended
        for agent_id, agent_step in terminal_steps.items():
            obs = agent_step.obs[0]
            reward = agent_step.reward
            data.append({
                "episode": episode,
                "step": step_count,
                "agent_id": agent_id,
                "observation": obs.tolist(),
                "action": None,  # last action already executed
                "reward": reward,
                "done": True
            })

        step_count += 1
        if len(terminal_steps) > 0:
            break  # episode finished

df = pd.DataFrame(data)
df.to_csv("3dball_data.csv", index=False)

env.close()

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

print("="*40, "System Information", "="*40)
uname = platform.uname()
print(f"System: {uname.system}")
print(f"Node Name: {uname.node}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

# Boot Time
print("="*40, "Boot Time", "="*40)
boot_time_timestamp = psutil.boot_time()
bt = datetime.fromtimestamp(boot_time_timestamp)
print(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

# let's print CPU information
print("="*40, "CPU Info", "="*40)
# number of cores
print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores:", psutil.cpu_count(logical=True))
# CPU frequencies
cpufreq = psutil.cpu_freq()
print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
# CPU usage
print("CPU Usage Per Core:")
for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
    print(f"Core {i}: {percentage}%")
print(f"Total CPU Usage: {psutil.cpu_percent()}%")

# Memory Information
print("="*40, "Memory Information", "="*40)
# get the memory details
svmem = psutil.virtual_memory()
print(f"Total: {get_size(svmem.total)}")
print(f"Available: {get_size(svmem.available)}")
print(f"Used: {get_size(svmem.used)}")
print(f"Percentage: {svmem.percent}%")
print("="*20, "SWAP", "="*20)
# get the swap memory details (if exists)
swap = psutil.swap_memory()
print(f"Total: {get_size(swap.total)}")
print(f"Free: {get_size(swap.free)}")
print(f"Used: {get_size(swap.used)}")
print(f"Percentage: {swap.percent}%")


# Disk Information
print("="*40, "Disk Information", "="*40)
print("Partitions and Usage:")
# get all disk partitions
partitions = psutil.disk_partitions()
for partition in partitions:
    print(f"=== Device: {partition.device} ===")
    print(f"  Mountpoint: {partition.mountpoint}")
    print(f"  File system type: {partition.fstype}")
    try:
        partition_usage = psutil.disk_usage(partition.mountpoint)
    except PermissionError:
        # this can be catched due to the disk that
        # isn't ready
        continue
    print(f"  Total Size: {get_size(partition_usage.total)}")
    print(f"  Used: {get_size(partition_usage.used)}")
    print(f"  Free: {get_size(partition_usage.free)}")
    print(f"  Percentage: {partition_usage.percent}%")
# get IO statistics since boot
disk_io = psutil.disk_io_counters()
print(f"Total read: {get_size(disk_io.read_bytes)}")
print(f"Total write: {get_size(disk_io.write_bytes)}")

