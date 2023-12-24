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
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class Arm_env(gym.Env):

    def __init__(self, para_dict, knolling_para=None):
        super(Arm_env, self).__init__()

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.init_offset_range = para_dict['init_offset_range']
        self.num_boxes = self.para_dict['boxes_num']
        self.box_range = self.para_dict['box_range']
        self.urdf_path = para_dict['urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.save_img_flag = para_dict['save_img_flag']

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
        self.x_manual_id = p.addUserDebugParameter("ee_x:", self.x_low_obs, self.x_high_obs, para_dict['reset_pos'][0])
        self.y_manual_id = p.addUserDebugParameter("ee_y:", self.y_low_obs, self.y_high_obs, para_dict['reset_pos'][1])
        self.z_manual_id = p.addUserDebugParameter("ee_z:", self.z_low_obs, self.z_high_obs, para_dict['reset_pos'][2])
        self.yaw_manual_id = p.addUserDebugParameter("ee_yaw:", -np.pi/2, np.pi/2, para_dict['reset_ori'][2])

        # Define action space (x, y, z, yaw)

        # self.action_space = spaces.Box(low=np.array([self.init_pos_range[0][0], self.init_pos_range[1][0], 0.00, -np.pi/2]),
        #                                high=np.array([self.init_pos_range[0][1],self.init_pos_range[1][1], 0.005, np.pi/2]),
        #                                dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-1, -1,  0.005, -np.pi/2]),
                                       high=np.array([1,1, 0.005, np.pi/2]),
                                       dtype=np.float32)

        # Define observation space (assuming a fixed number of objects for simplicity)
        num_objects = para_dict['boxes_num']
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(num_objects* 7+4,),  # Position (3) + Orientation (4) for each object
                                            dtype=np.float32)
        self.boxes_index =[]
        self.max_steps = 6
        self.create_scene()
        self.create_arm()
        self.reset()

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
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])

    def get_data_virtual(self):

        length_range = np.round(np.random.uniform(self.box_range[0][0], self.box_range[0][1], size=(self.num_boxes, 1)), decimals=3)
        width_range = np.round(np.random.uniform(self.box_range[1][0], np.minimum(length_range, 0.036), size=(self.num_boxes, 1)), decimals=3)
        height_range = np.round(np.random.uniform(self.box_range[2][0], self.box_range[2][1], size=(self.num_boxes, 1)), decimals=3)
        lwh_list = np.concatenate((length_range, width_range, height_range), axis=1)
        return lwh_list

    def create_objects(self,offline=True):

        if not offline:
            self.lwh_list = self.get_data_virtual()
            self.num_boxes = np.copy(len(self.lwh_list))
            rdm_ori_roll  = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.num_boxes, 1))
            rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.num_boxes, 1))
            rdm_ori_yaw   = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.num_boxes, 1))
            objs_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)

            rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.num_boxes, 1))
            rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.num_boxes, 1))
            rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.num_boxes, 1))
            x_offset = np.random.uniform(self.init_offset_range[0][0], self.init_offset_range[0][1])
            y_offset = np.random.uniform(self.init_offset_range[1][0], self.init_offset_range[1][1])
            objs_pos = np.concatenate((rdm_pos_x + x_offset, rdm_pos_y + y_offset, rdm_pos_z), axis=1)

            for i in range(self.num_boxes):
                obj_name = f'object_{i}'
                create_box(obj_name, objs_pos[i], p.getQuaternionFromEuler(objs_ori[i]), size=self.lwh_list[i])
                self.boxes_index.append(int(i + 2))
            for _ in range(100):
                p.stepSimulation()
            pos_ori_data = self.get_obs()[:-4].reshape(self.num_boxes,-1)
            np.savetxt('urdf/objs_location.csv', np.hstack([pos_ori_data[:,:7], self.lwh_list]))


        else:
            obj_info = np.loadtxt('urdf/objs_location.csv')
            objs_pos, objs_ori, self.lwh_list = obj_info[:,:3],obj_info[:,3:7],obj_info[:,7:10]
            for i in range(self.num_boxes):
                obj_name = f'object_{i}'
                create_box(obj_name, objs_pos[i], objs_ori[i], size=self.lwh_list[i])
                self.boxes_index.append(int(i + 2))


        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])


    def reset(self, seed=None, return_observation=True):


        home_loc = np.concatenate([self.para_dict['reset_pos'],self.para_dict['reset_ori'][2:]])
        self.act(a=home_loc)
        self.last_action = self.action

        if len(self.boxes_index) == 0:
            self.create_objects(offline=True)
        else:
            for k in self.boxes_index: p.removeBody(k)
            self.boxes_index = []
            self.create_objects(offline=True)

        for i in range(30):

            # traget_end = p.readUserDebugParameter(self.ee_manual_id)
            traget_end = 0.035
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                    targetPosition=traget_end, force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                    targetPosition=traget_end, force=self.para_dict['gripper_force'])
            p.stepSimulation()

        self.current_step = 0

        if seed is not None:
            np.random.seed(seed)


        initial_observation = self.get_obs()
        return initial_observation, {}


    def act(self, a):

        a[0] = (a[0] +1 )/2 * (self.x_high_obs - self.x_low_obs)+self.x_low_obs
        a[1] = (a[1] +1 )/2 * (self.y_high_obs - self.y_low_obs)+self.y_low_obs
        self.action = a
        target_location = [a[:3],[0,np.pi/2,a[3]]]

        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=target_location[0],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(
                                                      target_location[1]))
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=20)

        for _ in range(80):
            p.stepSimulation()
            # if self.is_render:
            #     time.sleep(1 / 480)

    def get_r(self,obs_list):

        obs_list = obs_list[:-4].reshape(self.num_boxes,-1) # last 4 data is the action
        # print('obs_list:',obs_list)
        dist = []
        # calculate the dist of each two objects
        for j in range(len(obs_list)-1):
            for i in range(j+1, len(obs_list)):
                dist.append(np.sqrt(np.sum((obs_list[j][:2]-obs_list[i][:2])**2)))

        dist_mean = np.mean(dist)
        if dist_mean>0.05:
            reward = 0.1
        else:
            reward = dist_mean

        # energy penalty:
        energy = np.sum((self.action-self.last_action)**2) * 0.01
        self.last_action = self.action
        reward -= energy

        # out-of-boundary penalty:
        if (self.x_low_obs<obs_list[:,0]).all() and \
                (self.x_high_obs > obs_list[:, 0]).all() and \
            (self.y_low_obs<obs_list[:,1]).all() and \
                (self.y_high_obs >obs_list[:,1]).all():

            return reward, False

        else: return -100, True

    def step(self,a):
        self.act(a)
        obs_list = self.get_obs()
        r,Done = self.get_r(obs_list)

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
        return obs_list, r, Done, truncated, {}


    def get_obs(self):
        obs_list = []
        for item in self.boxes_index:
            loc = p.getBasePositionAndOrientation(item)
            loc = np.concatenate(loc)
            obs_list.append(loc)
        obs_list = np.asarray(obs_list,dtype=np.float32).reshape(-1)
        obs_list = np.asarray(np.concatenate((obs_list,self.action)),dtype=np.float32)
        return obs_list

        # (width, length, image, image_depth, seg_mask) = p.getCameraImage(width=640,
        #                                                                  height=480,
        #                                                                  viewMatrix=self.view_matrix,
        #                                                                  projectionMatrix=self.projection_matrix,
        #                                                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # img = np.asarray(image).reshape((480,640,4))[...,:3]/255
        # # img = np.transpose(img,(1,2,0))
        #
        # img_path = self.para_dict['dataset_path'] + 'image.png'
        # plt.imsave(img_path, img)


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    para_dict = {'reset_pos': np.array([-0.9, 0, 0.005]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(2,3),
                 'is_render': True,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_force': 3,
                 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': './knolling_box/',
                 'urdf_path': './urdf/',}

    os.makedirs(para_dict['dataset_path'], exist_ok=True)
    env = Arm_env(para_dict=para_dict)


    for i in range(10000):
        # loc = [[0.15,0,0.015],[0,np.pi/2,0]]
        x_ml = p.readUserDebugParameter(env.x_manual_id)
        y_ml = p.readUserDebugParameter(env.y_manual_id)
        z_ml = p.readUserDebugParameter(env.z_manual_id)
        yaw_ml = p.readUserDebugParameter(env.yaw_manual_id)

        # random sample
        random_sample = env.action_space.sample()

        obs,r,done,_,_ = env.step(random_sample)
        print(r)

        if done:
            env.reset()

