from environment import *
import wandb
import argparse


np.random.seed(0)
random.seed(0)

def eval_model(num_episodes = 40):

    total_rewards = []

    for episode in range(num_episodes):
        obs,_ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # action[2] -= 0.003  # wrap the action space to make the model output 0.002
            print(action)

            obs, reward, done, _, info = env.step(action)
            print(reward)

            total_reward += reward
        print(total_reward)
        total_rewards.append(total_reward)

    average_reward = np.mean(total_rewards)
    print("Average Reward:", average_reward)
    return average_reward

para_dict = {'reset_pos': np.array([-0.9, 0, 0.005]), 'reset_ori': np.array([0, np.pi / 2, 0]),
             'save_img_flag': True,
             'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
             'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
             'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
             'boxes_num': 2,
             'boxes_num_max': 5,
             'is_render': True,
             'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
             'box_mass': 0.1,
             'gripper_force': 3,
             'move_force': 3,
             'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
             'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
             'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
             'dataset_path': './knolling_box/',
             'urdf_path': './urdf/', }

train_RL = False


run_id = '16'
max_num_obj = para_dict['boxes_num_max']
loggerID = 16

RLmode = "SAC"
num_scence = 200
Two_obs_Flag = False
log_path = f"logger/{RLmode}_{max_num_obj}objobs/log{loggerID}/"

env = Arm_env(para_dict=para_dict, init_scene=num_scence, two_obj_obs=Two_obs_Flag)

model = SAC.load(log_path + "/ppo_model_best.zip")

# Evaluate the model
eval_model()