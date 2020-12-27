import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time, random, os

class Env(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self, xml_directory, name, lengths, axes, dampings, frictions, gears, max_dof=None):
		
		self.name = name
		self.lengths = np.array(lengths)
		self.axes = np.array(axes)
		self.dampings = np.array(dampings)
		self.frictions = np.array(frictions)
		self.gears = np.array(gears)
		self.max_dof = np.array(max_dof)
		
		if max_dof:
			self.num_dummy = max_dof - len(self.lengths)-1
			
		self.distance_threshold = 0.05
		self._max_episode_steps = 50
		self.dummy_link = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
		
		mujoco_env.MujocoEnv.__init__(self, os.getcwd() + "/" + xml_directory + "/" + name + ".xml", 2)
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

class make_model_xml:
	def __init__(self, name):
		self.name = name
		self.file_data = ""
		self.num_links = 0
		self.gears = []
		self.lengths = []
		self.num_links = 0

		self.file_data += """<mujoco model="manipulator">
		<compiler angle="degree" coordinate="local" inertiafromgeom="auto" meshdir="meshes/"/>
		<option gravity="0 0 0" integrator="RK4" iterations="30" timestep="0.01"/>

		<default>
		<joint armature="1" limited="true" stiffness="10" damping="15" frictionloss="5.0"/>
		<geom conaffinity="1" condim="3" contype="1" friction=".5 .1 .1" margin="0.002"/>
		<motor ctrllimited="true"/>
		<position ctrllimited="true"/>
		</default>

		<asset>
		<texture builtin="gradient" height="100" rgb1="1.0 1.0 1.0" rgb2="1.0 1.0 1.0" type="skybox" width="100"/>
		<texture builtin="flat" height="32" name="arm_tex" rgb1="0.9 0.3 0.3" type="2d" width="32"/>
		<texture builtin="flat" height="32" name="gripper_tex" rgb1="0.0 0.0 0.0" type="cube" width="32"/>

		<material name="arm_mat" shininess="0.1" specular="0.1" texture="arm_tex"/>
		<material name="gripper_mat" shininess="1" specular="1" texture="gripper_tex"/>
		</asset>

		<worldbody>
		<light cutoff="45" diffuse=".8 .8 .8" dir="0 0.16 -1" directional="false" name="light" pos="0 -2.0 8.0" specular="0.3 0.3 0.3"/>

		<body name="base_link" pos="0 0 -0.1">
		<geom fromto="0 0 0 0 0 0.1" material="arm_mat" name="base_link" rgba="0.0 0.0 0.0 1" size="0.085" type="cylinder"/>
		</body>

		"""

	def add_link(self, link_length, axis, range_limit, damping, friction, gear, last=False):
		self.num_links += 1

		try:
			if self.num_links > 2:
				self.file_data += '<body name="l{}" pos="{} {} 0">\n'.format(self.num_links, self.lengths[-1], (-1)**(self.num_links-1)*0.16)
				self.file_data += '<geom fromto="0 0 0 {} 0 0" material="arm_mat" name="l{}" rgba="0.9 0.3 0.3 1" size="0.085" type="cylinder"/>\n'.format(link_length, self.num_links)
				if not last:
					self.file_data += '<geom fromto="{} 0 0 {} {} 0" material="arm_mat" name="l{}_ex" rgba="0.0 0.0 0.0 1" size="0.085" type="capsule"/>\n'.format(link_length, link_length, (-1)**(self.num_links)*0.16, self.num_links)
			else:
				self.file_data += '<body name="l{}" pos="0 {} {}">\n'.format(self.num_links, (-1)**(self.num_links-1)*0.16, self.lengths[-1])
				self.file_data += '<geom fromto="0 0 0 {} 0 0" material="arm_mat" name="l{}" rgba="0.9 0.3 0.3 1" size="0.085" type="cylinder"/>\n'.format(link_length, self.num_links)
				self.file_data += '<geom fromto="{} 0 0 {} {} 0" material="arm_mat" name="l{}_ex" rgba="0.0 0.0 0.0 1" size="0.085" type="capsule"/>\n'.format(link_length, link_length, (-1)**(self.num_links)*0.16, self.num_links)

		except IndexError:
			self.file_data += '<body name="l{}" pos="0 0 0">\n'.format(self.num_links)
			self.file_data += '<geom fromto="0 0 0 0 0 {}" material="arm_mat" name="l{}" rgba="0.9 0.3 0.3 1" size="0.085" type="cylinder"/>\n'.format(link_length, self.num_links)
			self.file_data += '<geom fromto="0 0 {} 0 {} {}" material="arm_mat" name="l{}_ex" rgba="0.0 0.0 0.0 1" size="0.085" type="capsule"/>\n'.format(link_length, (-1)**(self.num_links)*0.16, link_length, self.num_links)

		if self.num_links == 1:
			axis = [0, 0, 1]
		self.file_data += '<joint axis="{} {} {}" name="j{}" pos="0 0 0" range="{} {}" damping="{}" frictionloss="{}"/>\n'.format(axis[0],axis[1],axis[2],self.num_links, range_limit[0], range_limit[1],damping,friction)
		self.file_data += "\n"

		self.lengths.append(link_length)
		self.gears.append(gear)

	def save_file(self, directory):

		self.file_data += '<body name="gripper" pos="{} 0 0">\n'.format(self.lengths[-1])
		self.file_data += '<geom fromto="0.05 -0.07 0 0.05 0.07 0" material="gripper_mat" name="gripper" rgba="0.0 0.0 0.0 1" size="0.085" type="capsule"/>'
		self.file_data += '<joint axis="1 0 0" name="j{}" pos="0 0 0" range="-90 90" damping="15" frictionloss="5.0"/>'.format(self.num_links+1)

		self.file_data += '<body name="finger" pos="0.1 0 0">\n'
		self.file_data += '<geom fromto="0 0.0 0 0.1 0.0 0" material="gripper_mat" name="finger" rgba="0.0 0.0 0.0 1" size="0.020 0.06 0.06" type="box"/>\n'
		self.file_data += '</body>\n'
		self.file_data += '<site name="current" pos="0.2 0. 0.0" rgba="0. 0. 1.0 1" size="0.02" type="sphere"/>\n'
						

		for i in range(self.num_links+1):
			self.file_data += "</body>\n"
		self.file_data += "</worldbody>\n<actuator>\n"

		for i in range(1, self.num_links+1):
			self.file_data += '<motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="j{}" gear="{}"/>\n'.format(i, self.gears[i-1])

		self.file_data += '<motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="j{}" gear="{}"/>\n'.format(self.num_links+1, 80)

		self.file_data += "</actuator>\n</mujoco>"

		file = open(directory+"/"+self.name + ".xml", "w")
		file.write(self.file_data)
		file.close()

######################
######################
# Axes of link 1 is hardcoded in add_link method
######################
######################

class Create_envs:
	def __init__(self, robots_per_dof = 90, variance_limit = 0.25):
		#Settings seeds
		random.seed(1)
		np.random.seed(1)

		#Environment Data
		self.DOFs = [5,6,7]
		self.robots_per_dof = robots_per_dof
		self.standard_axis = [0.0, 1.0, 0.0]	#Except for joint 0 (base joint). which rotates about z axis
		self.standard_damping = 30.0
		self.standard_friction = 10.0
		self.standard_joint_gear = 80
		self.standard_length = 0.3
		self.joint_ranges = [[-90,90], [-90,60]] + [[-30,60]] * (max(self.DOFs)-3)
		self.variance_limit = variance_limit
		self.xml_directory = "Model_xmls"

		if not os.path.isdir(self.xml_directory):
			os.mkdir(self.xml_directory)

	def randomize(self, quantity):
		return round(quantity + random.uniform(-self.variance_limit*quantity, self.variance_limit*quantity),2)

	def create(self, padded=True):
		self.envs = []
		for DOF in self.DOFs:
			for robot_num in range(1, self.robots_per_dof+1):
				lengths, axes, dampings, frictions, gears = [],[],[],[],[]
				name = "Manipulator_" + str(DOF*100 + robot_num)

				if self.variance_limit > 0.25:
					name = "Manipulator_" + str(DOF*100 + robot_num) + "_" + str(self.variance_limit)

				xml = make_model_xml(name)
				num_links = DOF-1	#Ignoring the gripper joint/link as it does not affect target reaching

				for link_number in range(num_links):
					
					if link_number == 1:
						length = self.randomize(0.5)
					else:
						length = self.randomize(self.standard_length)
					
					axis = self.standard_axis
					#axis = self.randomize(self.standard_axis)
					damping = self.randomize(self.standard_damping)
					friction = self.randomize(self.standard_friction)
					gear = self.randomize(self.standard_joint_gear)
					xml.add_link(length, axis, self.joint_ranges[link_number], damping, friction, gear, link_number==num_links-1)

					lengths.append([length])
					axes.append(axis if link_number else [0,0,1])
					dampings.append([damping])
					frictions.append([friction])
					gears.append([gear])

				data = "Name: {}\n\tLengths: {}\n\tAxes: {}\n\tDampings: {}\n\tFrictions: {}\n\tGears: {}\n\n".format(name, lengths,axes,dampings,frictions,gears)
				f = open(self.xml_directory + "/env_data.txt", "a+")
				f.write(data)
				f.close()
				xml.save_file(self.xml_directory)
				
				if padded:
					env = Env(self.xml_directory, name, lengths, axes, dampings, frictions, gears, max_dof=max(self.DOFs))
				else:
					env = Env(self.xml_directory, name, lengths, axes, dampings, frictions, gears)
				
				self.envs.append(env)

		return self.envs

class Envs:

	def __init__(self, variance_limit=0.25):
		self.variance_limit = variance_limit

		if self.variance_limit == 0.25:
			envs_creator = Create_envs()
			self.envs = envs_creator.create()
			self.testing_envs = [env for i, env in enumerate(self.envs) if ((i+1)%envs_creator.robots_per_dof > envs_creator.robots_per_dof-10) or not ((i+1)%envs_creator.robots_per_dof)]
			self.training_envs = [env for i, env in enumerate(self.envs) if ((i+1)%envs_creator.robots_per_dof <= envs_creator.robots_per_dof-10) and ((i+1)%envs_creator.robots_per_dof)]
	
		else:
			envs_creator = Create_envs(robots_per_dof=10, variance_limit=self.variance_limit)
			self.envs = envs_creator.create()

	def get_envs(self):
		if self.variance_limit == 0.25:
			return Envs_subset(self.training_envs), Envs_subset(self.testing_envs)
		else:
			return Envs_subset(self.envs)

class Envs_subset:

	def __init__(self, env_list):
		self.envs = env_list

	def reset(self):
		observations = []
		desired_goals = []
		achieved_goals = []
			
		for env in self.envs:
			state = env.reset()
			observations.append(state['observation'])
			desired_goals.append(state['desired_goal'])
			achieved_goals.append(state['achieved_goal'])

		return {'observation': np.array(observations), "desired_goal": np.array(desired_goals), "achieved_goal": np.array(achieved_goals)}

	def step(self, actions):
		observations = []
		desired_goals = []
		achieved_goals = []
		infos = []
			
		for i, env in enumerate(self.envs):
			action = actions[i][:int(env.name[12])]
			state,_,_,info = env.step(action)
			observations.append(state['observation'])
			desired_goals.append(state['desired_goal'])
			achieved_goals.append(state['achieved_goal'])
			infos.append(info['is_success'])

		return {'observation': np.array(observations), "desired_goal": np.array(desired_goals), "achieved_goal": np.array(achieved_goals)}, {"is_success": infos}

		Manipulator_
