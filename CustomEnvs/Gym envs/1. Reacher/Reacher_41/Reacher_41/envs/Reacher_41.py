import numpy as np
import math, random
from gym import utils
from gym.envs.mujoco import mujoco_env

class model_data:

    def __init__(self):
        self.num_links = 4
        self.link_lengths = [0.13,0.14,0.07, 0.07]

    def get_goal(self):
        goal = np.array([0.00,-0.00])
        for i in range(len(self.link_lengths)):
            angle = random.random() * math.pi * 2
            link_length = self.link_lengths[i]
            if i == len(self.link_lengths) - 1:
                link_length = link_length+0.01
            shift_for_link = link_length * np.array([math.cos(angle), math.sin(angle)])
            goal = goal + shift_for_link
        return goal

modelData = model_data()

class Reacher_v41(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, add_dummy=False, max_links=5):

        self.add_dummy = add_dummy
        self.max_links = max_links
        if self.add_dummy:
            self.dummy_link = [0.0,0.0,0.0]

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'Reacher_41.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()/4
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        self.goal = modelData.get_goal()
        qpos[-2:] = self.goal
        qvel = self.init_qvel
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):

        #Extracting joint_pos and joint_vel:
        joint_pos = []
        joint_vel = []
        for link_number in range(modelData.num_links):
            joint_qpos_addr = self.model.get_joint_qpos_addr("joint" + str(link_number))
            joint_qvel_addr = self.model.get_joint_qvel_addr("joint" + str(link_number))
            joint_pos.append(float(self.data.qpos[joint_qpos_addr]))
            joint_vel.append(float(self.data.qvel[joint_qvel_addr]))
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel

        #Preparing observations
        obs = []
        obs.append(self.get_body_com("target")[0]) #Target x
        obs.append(self.get_body_com("target")[1]) #Target y
        
        for link_number in range(modelData.num_links):
            obs.append(self.joint_pos[link_number])
            obs.append(self.joint_vel[link_number])
            obs.append(modelData.link_lengths[link_number])

        if self.add_dummy:
            for _ in range(self.max_links - modelData.num_links):
                obs.extend(self.dummy_link)

        obs = np.array(obs)

        return obs