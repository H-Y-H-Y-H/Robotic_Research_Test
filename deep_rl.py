from environment import *

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
            print(action)

            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)

    average_reward = np.mean(total_rewards)
    print("Average Reward:", average_reward)
    return average_reward

train_RL = True
loggerID=1

para_dict = {'reset_pos': np.array([-0.9, 0, 0.005]), 'reset_ori': np.array([0, np.pi / 2, 0]),
             'save_img_flag': True,
             'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
             'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
             'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
             'boxes_num': np.random.randint(2, 3),
             'boxes_num_max': 8,
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

os.makedirs(para_dict['dataset_path'], exist_ok=True)
os.makedirs('log%d'%loggerID, exist_ok=True)

if train_RL:
    para_dict['is_render'] = False
    env = Arm_env(para_dict=para_dict)

    num_epoch = 10000

    # start from scratch
    # model = PPO("MlpPolicy", env, verbose=1)

    # pre-trained model:
    model = PPO.load("pre_trained/log1/ppo_model_best.zip")
    model.set_env(env)

    r_list = []
    r_max = -np.inf
    for epoch in range(num_epoch):
        model.learn(total_timesteps=10000)
        r = eval_model()
        r_list.append(r)

        if r>r_max:
            r= r_max
            # Save the model
            model.save("log%d/ppo_model_best.zip"%loggerID)
        model.save("log%d/ppo_model_last.zip"%loggerID)
        np.savetxt('log%d/r_logger.csv'%loggerID,r_list)

else:

    env = Arm_env(para_dict=para_dict)

    # Load the trained model
    model = PPO.load("pre_trained/log%d/ppo_model_best.zip"%loggerID)

    # Evaluate the model
    eval_model()

