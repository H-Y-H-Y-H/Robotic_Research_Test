from utils import *
import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import random
import time
import gymnasium as gym
from gymnasium import spaces
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env


class Arm_env(gym.Env):

    def __init__(self, para_dict, knolling_para=None, offline_data= True, init_scene = 40, init_num_obj=0,two_obj_obs=True):
        super(Arm_env, self).__init__()

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.init_offset_range = para_dict['init_offset_range']
        self.boxes_num = self.para_dict['boxes_num']
        self.box_range = self.para_dict['box_range']
        self.urdf_path = para_dict['urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.save_img_flag = para_dict['save_img_flag']

        # table  300 x 340
        self.x_low_obs = 0.03
        self.x_high_obs = 0.27
        self.y_low_obs = -0.14
        self.y_high_obs = 0.14
        self.z_low_obs = 0.0
        self.z_high_obs = 0.05
        x_grasp_accuracy = 0.2
        y_grasp_accuracy = 0.2
        z_grasp_accuracy = 0.2
        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy
        self.table_boundary = 0.03

        self.gripper_width = 0.024
        self.gripper_height = 0.034
        self.gripper_interval = 0.01

        self.angle_obj_limit = np.pi/3


        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
        else:
            p.connect(p.DIRECT)

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.150, 0, 0], #0.175
                                                               distance=0.4,
                                                               yaw=90,
                                                               pitch = -90,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=42,
                                                              aspect=640/480,
                                                              nearVal=0.1,
                                                              farVal=100)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(1. / 120.)


        self.ee_manual_id = p.addUserDebugParameter("ee_gripper:", 0, 0.035, 0.035)
        self.x_manual_id = p.addUserDebugParameter("ee_x:", -1, 1, para_dict['reset_pos'][0])
        self.y_manual_id = p.addUserDebugParameter("ee_y:", -1, 1, para_dict['reset_pos'][1])
        self.z_manual_id = p.addUserDebugParameter("ee_z:", self.z_low_obs, self.z_high_obs, para_dict['reset_pos'][2])
        self.yaw_manual_id = p.addUserDebugParameter("ee_yaw:", -np.pi/2, np.pi/2, para_dict['reset_ori'][2])

        # Define action space (x, y, z, yaw)
        self.action_space = spaces.Box(low=np.array([-1, -1,  0.005, -np.pi/2]),
                                       high=np.array([1,1, 0.005, np.pi/2]),
                                       dtype=np.float32)

        # Define observation space (assuming a fixed number of objects for simplicity)

        # observation: [x, y, theta, l, w] * N + ee 4
        self.two_obj_obs = two_obj_obs
        if self.two_obj_obs:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(5*2+ 4,),  # Position (3) + Orientation (4) for each object
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(5*2+ 4,),  # Position (3) + Orientation (4) for each object
                                                dtype=np.float32)

        self.traget_end = 0.035
        self.boxes_index =[]
        self.offline_data = offline_data
        self.max_steps = 3
        self.offline_scene_id = 0
        self.max_offline_scene_num = init_scene
        self.create_scene()
        self.create_arm()
        self.reset(fix_num_obj=init_num_obj) # fix_num_obj == 0: random 2-8 objects



    def create_scene(self,random_lightness=True,use_texture=True):
        self.baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useMaximalCoordinates=True)
        if random_lightness:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs+0.005],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs+0.005],
            lineWidth=10)
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs+0.005],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs+0.005],
            lineWidth=10)
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs+0.005],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs+0.005],
            lineWidth=10)
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs+0.005],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs+0.005],
            lineWidth=10)

        if use_texture:
            background = np.random.randint(1, 5)
            textureId = p.loadTexture(self.urdf_path + f"img_{background}.png")
            p.changeVisualShape(self.baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])
        p.setGravity(0, 0, -10)


    def create_arm(self):
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1_backup.urdf"),
                                 basePosition=[-0.08, 0, -0.005], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])

    def get_data_virtual(self):

        length_range = np.round(np.random.uniform(self.box_range[0][0], self.box_range[0][1], size=(self.boxes_num, 1)), decimals=3)
        width_range = np.round(np.random.uniform(self.box_range[1][0], np.minimum(length_range, 0.036), size=(self.boxes_num, 1)), decimals=3)
        height_range = np.round(np.random.uniform(self.box_range[2][0], self.box_range[2][1], size=(self.boxes_num, 1)), decimals=3)
        lwh_list = np.concatenate((length_range, width_range, height_range), axis=1)
        return lwh_list

    def create_objects(self): # a list of objects states

        if not self.offline_data: # random initialize the env for robot.
            while 1:
                self.lwh_list = self.get_data_virtual()
                rdm_ori_roll  = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.boxes_num, 1))
                rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.boxes_num, 1))
                rdm_ori_yaw   = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.boxes_num, 1))
                objs_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)

                rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.boxes_num, 1))
                rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.boxes_num, 1))
                rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.boxes_num, 1))
                x_offset = np.random.uniform(self.init_offset_range[0][0], self.init_offset_range[0][1])
                y_offset = np.random.uniform(self.init_offset_range[1][0], self.init_offset_range[1][1])
                objs_pos = np.concatenate((rdm_pos_x + x_offset, rdm_pos_y + y_offset, rdm_pos_z), axis=1)

                for i in range(self.boxes_num):
                    obj_name = f'object_{i}'
                    create_box(obj_name, objs_pos[i], p.getQuaternionFromEuler(objs_ori[i]), size=self.lwh_list[i])
                    self.boxes_index.append(p.getBodyUniqueId(int(i + 2)))
                for _ in range(100):
                    p.stepSimulation()

                obs_list,state_list = self.get_all_obj_info()

                # whether the objs are flip in the scene: state: x,y,z,r,p,y,l,w,h
                roll_list = state_list[:,3]
                pitch_list = state_list[:, 4]

                if (roll_list < -self.angle_obj_limit).any() or \
                   (roll_list > self.angle_obj_limit).any() or \
                   (pitch_list < -self.angle_obj_limit).any() or \
                   (pitch_list > self.angle_obj_limit).any():
                    for k in self.boxes_index: p.removeBody(k)
                    self.boxes_index = []

                    continue

                        # whether the objs are in the scene. observation: x,y,y,l,w
                if (self.x_low_obs<obs_list[:self.boxes_num, 0]).all() and \
                        (self.x_high_obs > obs_list[:self.boxes_num, 0]).all() and \
                    (self.y_low_obs<obs_list[:self.boxes_num, 1]).all() and \
                        (self.y_high_obs >obs_list[:self.boxes_num, 1]).all():
                    break
                else:
                    for k in self.boxes_index: p.removeBody(k)
                    self.boxes_index = []

            info_obj, state_list = self.get_all_obj_info()

        else:
            state_list = np.loadtxt('../obj_init_dataset/%dobj_%d.csv' % (self.boxes_num,self.offline_scene_id))

            objs_pos, objs_ori, self.lwh_list = state_list[:,:3],state_list[:,3:6],state_list[:,6:9]
            for i in range(self.boxes_num):
                obj_name = f'object_{i}'
                create_box(obj_name, objs_pos[i], p.getQuaternionFromEuler(objs_ori[i]), size=self.lwh_list[i])
                self.boxes_index.append(p.getBodyUniqueId(int(i + 2)))

            self.offline_scene_id += 1
            if self.offline_scene_id == self.max_offline_scene_num:
                self.offline_scene_id = 0

        return state_list

    def reset(self, seed=None, return_observation=True, fix_num_obj = 0):

        if fix_num_obj == 0:
            np.random.seed(seed)
            self.boxes_num = np.random.randint(2, 8)
        else:
            self.boxes_num = fix_num_obj

        home_loc = np.concatenate([self.para_dict['reset_pos'],self.para_dict['reset_ori'][2:]])
        self.act(input_a=home_loc)
        self.last_action = self.action

        for i in range(30):
            # traget_end = p.readUserDebugParameter(self.ee_manual_id)
             # end effector can close at this joint position.
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                    targetPosition=self.traget_end, force=10000)
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                    targetPosition=self.traget_end, force=10000)
            p.stepSimulation()

        while 1:
            if len(self.boxes_index) != 0:
                for k in self.boxes_index: p.removeBody(k)
                self.boxes_index = []

            self.create_objects()
            obs_list_flatten = self.get_obs()

            obs_list_all, state_list = self.get_all_obj_info()

            self.real_obs = np.copy(obs_list_flatten)

            # whether the objs are in the scene.
            if (self.x_low_obs<obs_list_all[:self.boxes_num, 0]).all() and \
                    (self.x_high_obs > obs_list_all[:self.boxes_num, 0]).all() and \
                (self.y_low_obs<obs_list_all[:self.boxes_num, 1]).all() and \
                    (self.y_high_obs > obs_list_all[:self.boxes_num, 1]).all():

                # # whether the objects are too closed to each other.
                # dist = []
                # # calculate the dist of each two objects
                # for j in range(self.boxes_num - 1):
                #     for i in range(j + 1, self.boxes_num):
                #         dist.append(np.sqrt(np.sum((obs_list[j][:2] - obs_list[i][:2]) ** 2)))
                # dist = np.array(dist)
                # if (dist<0.04).any():
                #     print('successful scene generated.')
                    break

        self.current_step = 0
        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])

        return obs_list_flatten, {}


    def act(self, input_a):

        a = np.copy(input_a)
        a[0] = (a[0] +1 )/2 * (self.x_high_obs - self.x_low_obs)+self.x_low_obs
        a[1] = (a[1] +1 )/2 * (self.y_high_obs - self.y_low_obs)+self.y_low_obs
        self.action = a
        a[2]-=0.003 # wrap the action space to make the model output 0.002
        target_location = [a[:3],[0,np.pi/2,a[3]]]

        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=target_location[0],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(
                                                      target_location[1]))
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=20)
        p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                targetPosition=self.traget_end, force=10000)
        p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                targetPosition=self.traget_end, force=10000)
        for _ in range(60):
            p.stepSimulation()
            if self.is_render:
                time.sleep(1 / 120)

    def get_r(self):

        obs_list,state_list = self.get_all_obj_info()

        # calculate the dist of each two objects
        dist = []
        for j in range(self.boxes_num-1):
            for i in range(j+1, self.boxes_num):
                dist.append(np.sqrt(np.sum((obs_list[j][:2]-obs_list[i][:2])**2)))

        dist_mean = np.mean(dist)
        if dist_mean>0.05:
            reward = 0.1
        else:
            reward = dist_mean

        # Penalty: Arm ee to the center of the pile.
        center_obj_loc = np.mean(obs_list[:self.boxes_num],axis=0)[:3]
        ee_obj = center_obj_loc - np.asarray(p.getLinkState(self.arm_id, 6)[0])
        ee_obj_r = -np.sum(ee_obj**2)
        reward += ee_obj_r

        # energy penalty:
        energy = np.sum((self.action-self.last_action)**2) * 0.01
        self.last_action = self.action
        reward -= energy

        # out-of-boundary penalty:
        if (self.x_low_obs<obs_list[:self.boxes_num,0]).all() and \
                (self.x_high_obs > obs_list[:self.boxes_num, 0]).all() and \
            (self.y_low_obs<obs_list[:self.boxes_num,1]).all() and \
                (self.y_high_obs >obs_list[:self.boxes_num,1]).all():

            return reward, False

        else: return -100, True

    def step(self,a):
        self.act(a)
        obs_list = self.get_obs()
        r, Done = self.get_r()

        # robot arm becomes crazy
        bar_pos = np.asarray(p.getLinkState(self.arm_id, 6)[0])
        if bar_pos[0] < 0:
            p.resetSimulation()
            self.boxes_index = []
            self.create_scene()
            self.create_arm()
            r = -100
            Done = True
            # print(a)

        self.current_step +=1
        # Check if maximum steps have been reached
        if self.current_step >= self.max_steps:
            Done = True

        truncated = False

        # use the initial observation for each episode.
        obs_list[:-4] = self.real_obs[:-4]

        return obs_list, r, Done, truncated, {}

    def get_all_obj_info(self):
        # Each object has 5 state: x,y,yaw + length, width.
        obs_list = []
        state_list = []
        for item_id in range(len(self.boxes_index)):

            urdf_id = self.boxes_index[item_id]
            pos,ori_q = p.getBasePositionAndOrientation(urdf_id)
            ori_eular = p.getEulerFromQuaternion(ori_q)
            obs = list(pos[:2]) + [ori_eular[2]] + list(self.lwh_list[item_id][:2])
            state = list(pos) + list(ori_eular)  + list(self.lwh_list[item_id])

            obs_list.append(obs)
            state_list.append(state)

        obs_list = sorted(obs_list, key=lambda obs_list: obs_list[0])
        state_list= sorted(state_list, key=lambda obs_list: obs_list[0])
        # order: based on the x axis.

        obs_list,state_list = np.asarray(obs_list),np.asarray(state_list)

        return obs_list,state_list

    def get_obs(self):
        # Each object has 5 state: x,y,yaw + length, width.

        obs_list, state_list = self.get_all_obj_info()

        if self.two_obj_obs:
            # calculate the dist of each two objects
            dist = []
            two_closed_obj_ID_list = []
            for j in range(self.boxes_num-1):
                for i in range(j+1, self.boxes_num):
                    dist.append(np.sqrt(np.sum((obs_list[j][:2]-obs_list[i][:2])**2)))
                    two_closed_obj_ID_list.append([j, i])

            # find the objects based on closest distance
            min_dist_id = np.argmin(dist)
            two_closed_obj_ID = two_closed_obj_ID_list[min_dist_id]

            # observation is two objects' info
            obs_list = obs_list[two_closed_obj_ID]
        else:
            obs_list = np.vstack((obs_list, np.zeros(((8-self.boxes_num),5))))

        obs_list = obs_list.reshape(-1)

        obs_list = np.asarray(np.concatenate((obs_list,self.action)),dtype=np.float32)

        return obs_list # flatten obj state list

        # (width, length, image, image_depth, seg_mask) = p.getCameraImage(width=640,
        #                                                                  height=480,
        #                                                                  viewMatrix=self.view_matrix,
        #                                                                  projectionMatrix=self.projection_matrix,
        #                                                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # img = np.asarray(image).reshape((480,640,4))[...,:3]/255
        # # img = np.transpose(img,(1,2,0))
        # img_path = self.para_dict['dataset_path'] + 'image.png'
        # plt.imsave(img_path, img)


if __name__ == '__main__':

    # np.random.seed(0)
    random.seed(0)
    #center of table: x = 0.12, y = 0
    para_dict = {'reset_pos': np.array([-0.9, 0, 0.005]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[-0.05, 0.05], [-0.05, 0.05], [0.01, 0.02]], 'init_offset_range': [[0.14, 0.16], [-0.01, 0.01]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': 8,
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_force': 3,
                 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': './knolling_box/',
                 'urdf_path': './urdf/',}

    num_scene = 10
    os.makedirs(para_dict['dataset_path'], exist_ok=True)


    MODE = 0 # RL random or Manual

    if MODE == 0:
        env = Arm_env(para_dict=para_dict, init_scene=num_scene,two_obj_obs=False)

        for i in range(10000):
            # random sample
            action = env.action_space.sample()
            obs,r,done,_,_ = env.step(action)
            print("Obs:", obs, "Reward:", r,'Action:',action)

            if done:
                env.reset(fix_num_obj=para_dict['boxes_num'])

    elif MODE == 1:
        env = Arm_env(para_dict=para_dict, init_scene=num_scene, offline_data=False, init_num_obj=para_dict['boxes_num'])

        n_samples = 10000
        count = 0

        while 1:
            # obj initialization:
            for k in env.boxes_index: p.removeBody(k)
            env.boxes_index = []
            info_obj = env.create_objects()

            # whether the objects are too closed to each other.
            dist = []
            # calculate the dist of each two objects
            for j in range(env.boxes_num - 1):
                for i in range(j + 1, env.boxes_num):
                    dist.append(np.sqrt(np.sum((info_obj[j][:2] - info_obj[i][:2]) ** 2)))
            dist = np.array(dist)
            if (dist < 0.035).any():
                print(f'successful scene generated {count}.')
                np.savetxt('../obj_init_dataset/%dobj_%d.csv' % (env.boxes_num, env.offline_scene_id), info_obj)
                env.offline_scene_id += 1
                count += 1
            if count == n_samples:
                break

    else:
        env = Arm_env(para_dict=para_dict, init_scene=num_scene, offline_data = False)

        for i in range(10000):
            # control robot arm manually:

            x_ml = p.readUserDebugParameter(env.x_manual_id)
            y_ml = p.readUserDebugParameter(env.y_manual_id)
            z_ml = p.readUserDebugParameter(env.z_manual_id)
            yaw_ml = p.readUserDebugParameter(env.yaw_manual_id)

            action = np.asarray([x_ml, y_ml, z_ml, yaw_ml])
            obs,r,done,_,_ = env.step(action)
            print(r)

