import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time

class model_data:

	def __init__(self):
		self.num_legs = 4
		self.leg_lengths = [[0.3, 0.6],[0.3, 0.6],[0.3, 0.3, 0.6],[0.3, 0.6]]
		self.leg_starts = [[0.25, 0.55],[0.25, -0.55],[-0.25, -0.55],[-0.25, 0.55]]
		self.joint_ranges = [[[-0.5234, 0.5234],[0.1745, 1.57]],[[-0.5234, 0.5234],[0.1745, 1.57]],[[-0.5234, 0.5234],[-0.5234, 0.5234],[0.1745, 1.57]],[[-0.5234, 0.5234],[0.1745, 1.57]]]
		self.joint_axes = [[[0, 0, 1],[0, 1, 0],[0, 1, 0]],[[0, 0, 1],[0, 1, 0]],[[0, 0, -1],[0, -1, 0],[0, -1, 0]],[[0, 0, -1],[0, -1, 0]]]
		self.joint_names = ['hip', 'ankle']

modelData = model_data()

def mod(vector):
	value = ((vector[0]**2 + vector[1]**2)**0.5)
	return value

class Quadruped_v23(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self, add_dummy=False, max_segments=3, max_legs=6):

		self.add_dummy = add_dummy
		self.max_legs = max_legs
		self.max_segments = max_segments
		if add_dummy:
			self.dummy_segment = [[0.0,0.0,0.0],[0.0,0.0],[0.0],[0.0]]
			self.dummy_leg = [[0.0,0.0]]
			for _ in range(max_segments):
				self.dummy_leg.extend(self.dummy_segment)

		mujoco_env.MujocoEnv.__init__(self, 'Quadruped_23.xml', 5)
		utils.EzPickle.__init__(self)
		
	def step(self, a, goal=[0, 1]):

		goal = np.array(goal)
		#Pervious xy pos and simulation step
		x_before = self.get_body_com("torso")[0]
		y_before = self.get_body_com("torso")[1]
		self.do_simulation(a, self.frame_skip)

		#Calculating reward
		x_after = self.get_body_com("torso")[0]
		y_after = self.get_body_com("torso")[1]

		x_movement = x_after - x_before
		y_movement = y_after - y_before
		distance_covered = ((x_movement**2) + (y_movement**2))**0.5
		
		speed = distance_covered / self.dt
		direction = [x_movement/distance_covered, y_movement/distance_covered]
		dot_product = sum(goal * np.array(direction))
		cos_theeta = dot_product / (mod(goal) * mod(direction))
		
		#Whether done
		state = self.state_vector()
		not_done = np.isfinite(state).all()	and state[2] >= 0.2
		done = not not_done
		ob = self._get_obs()

		movement_reward = speed * cos_theeta
		weighted_movement_reward = 2*movement_reward
		ctrl_cost = (.5 * np.square(a).sum())/(modelData.num_legs+0.5)
		survive_reward = 1.0
		if done:
			survive_reward = 0
		total_reward = weighted_movement_reward - ctrl_cost + survive_reward

		return ob, total_reward, done, dict(
			movement_reward= weighted_movement_reward,
			ctrl_reward = -ctrl_cost,
			survive_reward = survive_reward
			)

	def _get_obs(self):
		
		#Extracting joint_pos
		joint_pos = []
		for leg_number in range(modelData.num_legs):
			leg_joint_pos = []
			added_extra = False
			for joint in range(len(modelData.joint_names)):
				joint_qpos_addr = self.model.get_joint_qpos_addr(modelData.joint_names[joint]+ "_" + str(leg_number+1))
				joint_qpos = self.data.qpos[joint_qpos_addr]
				leg_joint_pos.append(float(joint_qpos))
				if leg_number == 2 and not added_extra:
					joint_qpos_addr = self.model.get_joint_qpos_addr('extra_joint_1')
					joint_qpos = self.data.qpos[joint_qpos_addr]
					leg_joint_pos.append(float(joint_qpos))
					added_extra = True
			joint_pos.append(leg_joint_pos)
		self.joint_pos = joint_pos

		#Preparing observations
		obs = []
		for leg_num in range(modelData.num_legs):
			if leg_num == 2:
				obs.append(modelData.leg_starts[leg_num])		#leg_start
				obs.append(modelData.joint_axes[leg_num][0]) 	#j1_range
				obs.append(modelData.joint_ranges[leg_num][0]) 	#j1_range
				obs.append([self.joint_pos[leg_num][0]])    	#j1_pos
				obs.append([modelData.leg_lengths[leg_num][0]])	#length_1
				obs.append(modelData.joint_axes[leg_num][1]) 	#j1_range
				obs.append(modelData.joint_ranges[leg_num][1])	#j2_range
				obs.append([self.joint_pos[leg_num][1]])		#j2_pos
				obs.append([modelData.leg_lengths[leg_num][1]])	#length_2
				obs.append(modelData.joint_axes[leg_num][2]) 	#j1_range
				obs.append(modelData.joint_ranges[leg_num][2])	#j2_range
				obs.append([self.joint_pos[leg_num][2]])		#j2_pos
				obs.append([modelData.leg_lengths[leg_num][2]])

			else:
				obs.append(modelData.leg_starts[leg_num])		#leg_start
				obs.append(modelData.joint_axes[leg_num][0]) 	#j1_range
				obs.append(modelData.joint_ranges[leg_num][0]) 	#j1_range
				obs.append([self.joint_pos[leg_num][0]])    	#j1_pos
				obs.append([modelData.leg_lengths[leg_num][0]])	#length_1
				obs.append(modelData.joint_axes[leg_num][1]) 	#j1_range
				obs.append(modelData.joint_ranges[leg_num][1])	#j2_range
				obs.append([self.joint_pos[leg_num][1]])		#j2_pos
				obs.append([modelData.leg_lengths[leg_num][1]])	#length_2

				if self.add_dummy:
					for _ in range(self.max_segments - len(modelData.leg_lengths[leg_num])):
						obs.extend(self.dummy_segment)
		
		if self.add_dummy:
			for _ in range(self.max_legs - modelData.num_legs):
				obs.extend(self.dummy_leg)
			
		obs = [j for i in obs for j in i]

		obs = np.array(obs)

		return obs

	def reset_model(self):
		qpos = self.init_qpos
		qvel = self.init_qvel

		self.set_state(qpos, qvel)
		return self._get_obs()

	def viewer_setup(self):
		self.viewer.cam.distance = self.model.stat.extent * 0.5