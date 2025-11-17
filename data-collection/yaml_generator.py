import os
import time
import random
import yaml
from typing import Dict, Tuple, Optional, List

# ============================================================================
# General recommended constants,
# as per https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Training-Configuration-File.html
# ============================================================================


# NOTE:
# I don't really get why we are using POCA, \
# as it's something we would more use for when we are training multiple interacting agents at once \
# (e.g. agents are collaborating/are being spawned and consumed). \
# POCA is basically PPO with a shared critic for multi-agent setups.

LEARNING_RATES = {
    'ppo': (1e-5, 3e-4),
    'sac': (3e-5, 3e-4)
}

# NOTE:
# great heuristic in ML to split ranges geometrically by powers of 2 rather than linearly
BATCH_SIZES = {
    'ppo_continuous': [512, 1024, 2048, 4096],
    'ppo_discrete': [32, 64, 128, 256, 512],
    'sac_continuous': [128, 256, 512, 1024],
    'sac_discrete': [32, 64, 128, 256]
}

BUFFER_SIZE_MULTIPLIERS = {
    # NOTE:
    # PPO/POCA are on-policy (only learn from recent experiences), \
    # so buffer size is multiple of batch size.
    'ppo': [20, 40, 80],  # buffer_size = batch_size * multiplier
    'sac': [50000, 100000, 200000, 500000] # Absolute values for SAC # NOTE: SAC is off-policy, so uses replay buffer
}

TIME_HORIZONS = [64, 128, 256, 512, 1024, 2048]

# NOTE:
# if it's not feasible to do a full network search on all network size combinations, \
# we could also define something like complexity classes to sample from?
# Also, SAC typically uses larger networks, so might want to bias towards 128+, 2+ layers
HIDDEN_UNITS = [32, 64, 128, 256, 512]
NUM_LAYERS = [1, 2, 3]
NORMALIZE_OPTIONS = [True, False]  # whether to normalize vector observations

GAMMA_VALUES = [0.9, 0.95, 0.99, 0.995]

# Visual encoding types (only matters if using camera/visual observations)
# NOTE: Only kept this here in case some of the projects use camera observations, \
# which is surely not the case for 3DBall and Soccer. \
# As such, I didn't use it in the generator code yet. Tell me if needed.
# VIS_ENCODE_TYPES = ['simple', 'nature_cnn', 'resnet', 'match3']

# Learning rate schedules
# NOTE: PPO can benefit from linear schedules, SAC typically uses constant
LR_SCHEDULES = ['linear', 'constant']

# ----------------------------------------------------------------------------
# PPO-specific hyperparameters:
# ----------------------------------------------------------------------------
PPO_BETA_RANGE = (1e-4, 1e-2)  # entropy bonus coefficient
PPO_EPSILON_RANGE = (0.1, 0.3)  # clipping parameter
PPO_LAMBDA_RANGE = (0.90, 0.95)  # GAE lambda for advantage estimation
PPO_NUM_EPOCH_OPTIONS = [3, 4, 5, 8, 10]  # number of passes through experience buffer

# ----------------------------------------------------------------------------
# SAC-specific hyperparameters:
# ----------------------------------------------------------------------------
SAC_TAU_RANGE = (0.005, 0.01)  # target network update rate
SAC_INIT_ENTCOEF_CONTINUOUS_RANGE = (0.05, 1.0)  # initial entropy coefficient for continuous actions
SAC_INIT_ENTCOEF_DISCRETE_RANGE = (0.05, 0.5)  # initial entropy coefficient for discrete actions
SAC_BUFFER_INIT_STEPS_OPTIONS = [0, 1000, 5000, 10000]  # random steps before learning starts
SAC_STEPS_PER_UPDATE_OPTIONS = [1, 2, 3, 4]  # how often to update the network
SAC_SAVE_REPLAY_BUFFER = True  # whether to save replay buffer #NOTE: costly in terms of disk space, but will thank in case of crashes
SAC_THREADING = [True, False]  # whether to enable threading for SAC #NOTE: SAC benefits from threading

# ----------------------------------------------------------------------------
# POCA-specific hyperparameters (same as PPO):
# ----------------------------------------------------------------------------
# POCA uses the same hyperparameters as PPO (beta, epsilon, lambda, num_epoch)
# since it's essentially PPO with a shared critic for multi-agent scenarios



# ============================================================================
# Configuration Generator Functions
# ============================================================================

def make_ppo_config(max_steps: int,
                   time_horizon: Optional[int] = None,
                   action_type: str = 'continuous',
                   enable_schedules: bool = True,
                   **kwargs) -> Dict:
    """
    Generate PPO training configuration with randomized hyperparameters.

    Args:
        max_steps: Total training steps
        time_horizon: Episode length (auto-generated if None)
        action_type: 'continuous' or 'discrete'
        complexity: 'low', 'medium', or 'high' network complexity
        enable_schedules: Whether to use learning rate schedules
        **kwargs: Override specific parameters

    Returns:
        Dictionary containing PPO configuration
    """
    batch_sizes = BATCH_SIZES['ppo_continuous'] if action_type == 'continuous' else BATCH_SIZES['ppo_discrete']
    batch_size = kwargs.get('batch_size', random.choice(batch_sizes))

    buffer_multiplier = kwargs.get('buffer_multiplier', random.choice(BUFFER_SIZE_MULTIPLIERS['ppo']))
    buffer_size = kwargs.get('buffer_size', batch_size * buffer_multiplier)

    if time_horizon is None:
        time_horizon = random.choice(TIME_HORIZONS)

    hidden_units = kwargs.get('hidden_units', random.choice(HIDDEN_UNITS))
    num_layers = kwargs.get('num_layers', random.choice(NUM_LAYERS))

    # PPO-specific hyperparameters
    learning_rate = kwargs.get('learning_rate',
                               round(random.uniform(*LEARNING_RATES['ppo']), 6))
    beta = kwargs.get('beta',
                     round(random.uniform(*PPO_BETA_RANGE), 6))
    epsilon = kwargs.get('epsilon',
                        round(random.uniform(*PPO_EPSILON_RANGE), 3))
    lambd = kwargs.get('lambd',
                      round(random.uniform(*PPO_LAMBDA_RANGE), 3))
    num_epoch = kwargs.get('num_epoch', random.choice(PPO_NUM_EPOCH_OPTIONS))
    gamma = kwargs.get('gamma', random.choice(GAMMA_VALUES))

    lr_schedule = random.choice(LR_SCHEDULES) if enable_schedules else 'constant'

    config = {
        "trainer_type": "ppo",
        "hyperparameters": {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "learning_rate": learning_rate,
            "beta": beta,
            "epsilon": epsilon,
            "lambd": lambd,
            "num_epoch": num_epoch,
            "learning_rate_schedule": lr_schedule,
            "beta_schedule": lr_schedule if enable_schedules else 'constant',
            "epsilon_schedule": lr_schedule if enable_schedules else 'constant',
        },
        "network_settings": {
            "normalize": kwargs.get('normalize', random.choice(NORMALIZE_OPTIONS)),
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            #"vis_encode_type": kwargs.get('vis_encode_type', 'simple'),
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": gamma,
                "strength": 1.0,
            }
        },
        "max_steps": max_steps,
        "time_horizon": time_horizon,
        "summary_freq": kwargs.get('summary_freq', random.choice([10000, 25000, 50000])),
        "checkpoint_interval": kwargs.get('checkpoint_interval', random.choice([100000, 250000, 500000])),
        "keep_checkpoints": kwargs.get('keep_checkpoints', 5),
        "threaded": kwargs.get('threaded', False),
    }

    return config


def make_sac_config(max_steps: int,
                   time_horizon: Optional[int] = None,
                   action_type: str = 'continuous',
                   **kwargs) -> Dict:
    """
    Generate SAC training configuration with randomized hyperparameters.

    SAC is typically used for continuous control and uses a replay buffer.

    Args:
        max_steps: Total training steps
        time_horizon: Episode length (auto-generated if None)
        action_type: 'continuous' or 'discrete'
        complexity: 'low', 'medium', or 'high' network complexity
        **kwargs: Override specific parameters

    Returns:
        Dictionary containing SAC configuration
    """
    batch_sizes = BATCH_SIZES['sac_continuous'] if action_type == 'continuous' else BATCH_SIZES['sac_discrete']
    batch_size = kwargs.get('batch_size', random.choice(batch_sizes))

    buffer_size = kwargs.get('buffer_size', random.choice(BUFFER_SIZE_MULTIPLIERS['sac']))

    if time_horizon is None:
        time_horizon = random.choice(TIME_HORIZONS)

    hidden_units = kwargs.get('hidden_units', random.choice(HIDDEN_UNITS))
    num_layers = kwargs.get('num_layers', random.choice(NUM_LAYERS))

    # SAC-specific hyperparameters
    learning_rate = kwargs.get('learning_rate',
                               round(random.uniform(*LEARNING_RATES['sac']), 6))
    tau = kwargs.get('tau',
                    round(random.uniform(*SAC_TAU_RANGE), 4))
    init_entcoef = kwargs.get('init_entcoef',
                              round(random.uniform(*SAC_INIT_ENTCOEF_CONTINUOUS_RANGE) if action_type == 'continuous'
                                    else random.uniform(*SAC_INIT_ENTCOEF_DISCRETE_RANGE), 3))
    gamma = kwargs.get('gamma', random.choice(GAMMA_VALUES))

    # Buffer and update settings
    buffer_init_steps = kwargs.get('buffer_init_steps', random.choice(SAC_BUFFER_INIT_STEPS_OPTIONS))
    steps_per_update = kwargs.get('steps_per_update', random.choice(SAC_STEPS_PER_UPDATE_OPTIONS))
    save_replay_buffer = kwargs.get('save_replay_buffer', SAC_SAVE_REPLAY_BUFFER)

    config = {
        "trainer_type": "sac",
        "hyperparameters": {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "learning_rate": learning_rate,
            "tau": tau,
            "steps_per_update": steps_per_update,
            "init_entcoef": init_entcoef,
            "buffer_init_steps": buffer_init_steps,
            "save_replay_buffer": save_replay_buffer,
            "learning_rate_schedule": "constant",
        },
        "network_settings": {
            "normalize": kwargs.get('normalize', random.choice(NORMALIZE_OPTIONS)),
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            #"vis_encode_type": kwargs.get('vis_encode_type', 'simple'),
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": gamma,
                "strength": 1.0,
            }
        },
        "max_steps": max_steps,
        "time_horizon": time_horizon,
        "summary_freq": kwargs.get('summary_freq', random.choice([10000, 25000, 50000])),
        "checkpoint_interval": kwargs.get('checkpoint_interval', random.choice([100000, 250000, 500000])),
        "keep_checkpoints": kwargs.get('keep_checkpoints', 5),
        "threaded": kwargs.get('threaded', random.choice(SAC_THREADING)),  # SAC benefits from threading
    }

    return config

# ============================================================================
# Main Generation Function
# ============================================================================

def generate_yaml(training_type: str,
                 max_steps: Optional[int] = None,
                 time_horizon: Optional[int] = None,
                 action_type: str = 'continuous',
                 output_dir: str = "generated",
                 behavior_name: str = "3DBall",
                 **kwargs) -> Tuple[bool, str]:
    """
    Generate a complete ML-Agents training configuration YAML file.

    This function creates diverse training configurations for data collection
    to model training resource usage patterns.

    Args:
        training_type: Algorithm type - 'ppo', 'sac', 'poca', or 'soc' (alias for 'sac')
        max_steps: Total training steps (default: random between 100k-2M)
        time_horizon: Episode length (default: randomized based on algorithm)
        num_parallel_envs: Number of parallel environments (impacts resources)
        action_type: 'continuous' or 'discrete'
        enable_env_settings: Include environment/engine settings
        output_dir: Output directory for generated YAML
        behavior_name: Name of the behavior in the config

    Returns:
        Tuple of (success: bool, path_or_error: str)

    Example:
        >>> success, path = generate_yaml('ppo', max_steps=500000,
        ...                               num_parallel_envs=4,
        ...                               complexity='high')
        >>> print(f"Generated: {path}")
    """
    try:
        # Normalize algorithm name
        algo = str(training_type).strip().lower()
        if algo in ('soc', 'sac'): # NOTE: Saw it in Sacha's code, seems harmless
            algo = 'sac'

        if algo not in ('ppo', 'sac'):
            return False, f"Invalid training type: {training_type}. Must be 'ppo' or 'sac'"

        # Validate inputs
        if max_steps is not None and max_steps < 1000:
            return False, "max_steps must be at least 1000"
        if max_steps is None:
            max_steps = random.choice([100000, 250000, 500000, 1000000, 2000000])

        config_builders = {
            "ppo": make_ppo_config,
            "sac": make_sac_config
        }

        behavior_config = config_builders[algo](
            max_steps=max_steps,
            time_horizon=time_horizon,
            action_type=action_type,
            **kwargs
        )

        full_config = {
            "behaviors": {
                behavior_name: behavior_config
            }
        }

        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        filename = f"{algo}_{timestamp}_{random_suffix}.yaml"
        output_path = os.path.join(output_dir, behavior_name + "_" + filename)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(full_config, f, sort_keys=False, default_flow_style=False)

        return True, output_path

    except Exception as e:
        return False, f"Error generating YAML: {str(e)}"


# ============================================================================
# Batch Generation Function
# ============================================================================

def generate_batch(algorithms: List[str],
                   behaviours: List[str] = ['3DBall','Crawler','GridFoodCollector','PushBlock'],
                  count_per_algorithm: int = 5,
                  output_dir: str = "generated",
                  **kwargs) -> List[Tuple[bool, str]]:
    """
    Generate multiple YAML files for different algorithms.

    Args:
        algorithms: List of algorithm names ('ppo', 'sac')
        count_per_algorithm: Number of configs to generate per algorithm
        output_dir: Output directory
        **kwargs: Additional parameters passed to generate_yaml

    Returns:
        List of (success, path_or_error) tuples
    """
    results = []

    for algo in algorithms:
        for behaviour in behaviours:
            for i in range(count_per_algorithm):
                success, path = generate_yaml(
                    training_type=algo,
                    output_dir=output_dir,
                    behavior_name=behaviour,
                    **kwargs
                )
                results.append((success, path))

                if success:
                    print(f"Generated {algo.upper()} config {i+1}/{count_per_algorithm}: {path}")
                else:
                    print(f"Failed to generate {algo.upper()} config {i+1}: {path}")

    return results

# ============================================================================
# test
# ============================================================================

if __name__ == "__main__":
    generate_batch(['ppo','sac'])
