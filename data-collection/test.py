import pandas as pd
from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name="3DBall/UnityEnvironment.exe")

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
