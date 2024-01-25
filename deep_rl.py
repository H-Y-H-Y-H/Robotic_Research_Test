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


loggerID= 14
RLmode = 'PPO' # "PPO"
num_scence = 10000
Two_obs_Flag = False

log_path = f"logger/{RLmode}_{para_dict['boxes_num_max']}objobs/log{loggerID}/"

os.makedirs(log_path, exist_ok=True)

if train_RL:
    wandb.init(project="RL_sep3", entity="robotics",name=str(loggerID)) # , mode="disabled"

    para_dict['is_render'] = False
    env = Arm_env(para_dict=para_dict, init_scene=num_scence, two_obj_obs=Two_obs_Flag)

    num_epoch = 100000

    # # # start from scratch
    # if RLmode == 'SAC':
    #     model = SAC("MlpPolicy", env, verbose=1)
    # elif RLmode == 'PPO':
    #     model = PPO("MlpPolicy", env, verbose=1)

    # # # # pre-trained model:
    # model = SAC.load(f"logger/SAC_{para_dict['boxes_num_max']}objobs/log{15}/ppo_model_best.zip")
    # model.set_env(env)

    # # # pre-trained model:
    model = PPO.load(f"logger/PPO_{para_dict['boxes_num_max']}objobs/log{14}/ppo_model_best.zip")
    model.set_env(env)

    # Configure wandb with hyperparameters
    config = {
        "loggerID": loggerID,
        'num_scence': num_scence,
        'Algorithm': RLmode,
        'max_num_obj': para_dict['boxes_num_max'],
        'obs_size': env.real_obs.shape,
        'Two_obs_Flag': Two_obs_Flag
    }
    wandb.config.update(config)

    r_list = []
    r_max = -np.inf
    for epoch in range(num_epoch):
        t0 = time.time()
        model.learn(total_timesteps=10000)
        r = eval_model(num_episodes=100)
        r_list.append(r)

        if r>r_max:
            r_max = r

            # Save the model
            model.save(log_path+"/ppo_model_best.zip")
        np.savetxt(log_path+"/r_logger.csv",r_list)
        t1 = time.time()

        # Log metrics to wandb
        wandb.log({"reward": r,
                   "best_r":r_max,
                   'epoch time used (s)': int(t1-t0),
                   'epoch:':epoch})

else:
    run_id = '15'
    api = wandb.Api()
    proj_name = 'RL_sep3'
    runs = api.runs("robotics/%s"%proj_name)
    config = None
    for run in runs:
        if run.name == run_id:
            print('loading configuration')
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    config = argparse.Namespace(**config)

    # summary = None
    # for run in runs:
    #     if run.name == run_id:
    #         print('loading configuration')
    #         summary = {k: v for k, v in run.summary.items() if not k.startswith('_')}
    # summary = argparse.Namespace(**summary)

    loggerID = config.loggerID

    RLmode = config.Algorithm
    num_scence = config.num_scence
    Two_obs_Flag = config.Two_obs_Flag
    log_path = f"logger/{RLmode}_{config.max_num_obj}objobs/log{loggerID}/"


    env = Arm_env(para_dict=para_dict, init_scene=num_scence, two_obj_obs=config.Two_obs_Flag)

    # Load the trained model
    if RLmode == 'PPO':
        model = PPO.load(log_path+"/ppo_model_best.zip")
    else:
        model = SAC.load(log_path+"/ppo_model_best.zip")


    # Evaluate the model
    eval_model()

