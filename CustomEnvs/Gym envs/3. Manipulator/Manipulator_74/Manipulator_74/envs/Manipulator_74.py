import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time

class Manipulator_v74(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self, max_dof=None):
		self.dummy_link = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
		self.lengths = [[0.37], [0.4], [0.23], [0.32], [0.27], [0.26]]
		self.axes = [[0, 0, 1], [0.08, 1.1, 0.06], [0.15, 0.77, 0.14], [0.12, 0.88, -0.12], [0.07, 0.92, 0.15], [-0.03, 1.14, 0.25]]
		self.dampings = [[24.29], [26.93], [25.35], [31.58], [30.64], [32.75]]
		self.frictions = [[10.5], [8.74], [9.69], [11.68], [8.87], [11.46]]
		self.gears = [[76.33], [89.98], [60.84], [68.26], [83.43], [92.35]]
		self.max_dof = max_dof
		if max_dof:
			self.num_dummy = max_dof - len(self.lengths) - 1
			
		self.distance_threshold = 0.05
		self._max_episode_steps = 50
		
		mujoco_env.MujocoEnv.__init__(self, 'Manipulator_74.xml', 2)
		utils.EzPickle.__init__(self)

	def step(self, action):
		self.do_simulation(action, self.frame_skip)
		obs = self._get_obs()
		achieved = self.sim.data.get_site_xpos("current")
		try:
			goal = self.goal.copy()
		except AttributeError:
			goal = np.random.uniform(0.0, 0.0, size=3)
		state = {"observation": obs, "achieved_goal": achieved, "desired_goal":goal}
		reward = self.compute_reward(achieved, goal)
		done = self._is_success(achieved, goal)
		return state, reward, done, {"is_success": float(done)}

	def _get_obs(self):
		qpos, qvel = self.sim.data.qpos[:-1], self.sim.data.qvel[:-1]
		qpos, qvel = qpos.reshape(qpos.shape[0],1), qvel.reshape(qvel.shape[0],1)
		obs = np.concatenate((self.lengths, self.axes, qpos, qvel, self.dampings, self.frictions, self.gears),1)
		if not self.max_dof:
			return obs.flatten()
		else:
			dummy = self.dummy_link.repeat(self.num_dummy, axis=0)
			obs = np.concatenate((obs, dummy))
			return obs

	def _is_success(self, desired_goal, achieved_goal):
		assert desired_goal.shape == achieved_goal.shape
		d = np.linalg.norm(desired_goal - achieved_goal, axis=-1)
		return (d < self.distance_threshold).astype(np.float32)

	def compute_reward(self, achieved, goal, info=None):
		assert goal.shape == achieved.shape
		d = np.linalg.norm(goal - achieved, axis=-1)
		return -(d > self.distance_threshold).astype(np.float32)

	def reset_model(self):
		qpos = self.init_qpos
		qvel = self.init_qvel
		self.set_state(qpos, qvel)
		
		obs = self._get_obs()
		self.goal = self._sample_goal()
		achieved = self.sim.data.get_site_xpos("current")
		state = {"observation": obs, "achieved_goal": achieved, "desired_goal":self.goal}
		return state

	def _sample_goal(self):
		goal = self.sim.data.get_site_xpos("current")
		goal[0] = goal[0] - 0.25
		goal = goal + np.random.uniform(-0.15, 0.15, size=3)
		return goal.copy()

	def viewer_setup(self):
		self.viewer.cam.distance = self.model.stat.extent * 2.5