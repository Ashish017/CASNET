import random, numpy as np

random.seed(1)
np.random.seed(1)

variance_limit = 0.25

DOFs = [5,6,7]
robots_per_dof = 10
Standard_axis = [0.0, 1.0, 0.0]	#Except for joint 0 (base joint). which rotates about z axis
Standard_damping = 30.0
Standard_friction = 10.0
Standard_joint_gear = 80
joint_ranges = [[-90,90], [-90,60], [-30,60], [-30,60], [-30,60], [-30,60]]

show_data = True

def randomize(quantity):
	try:
		data = list(quantity + np.random.uniform(-0.25, 0.25, size=len(quantity)))
		for i, data_point in enumerate(data):
			data[i] = round(data_point, 2)
		return data
	except TypeError:
		return round(quantity + random.uniform(-0.25*quantity, 0.25*quantity),2)

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

	def save_file(self):

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

		file = open(self.name + ".xml", "w")
		file.write(self.file_data)
		file.close()

######################
######################
# Axes of link 1 is hardcoded in add_link method
######################
######################

def create_xmls(show_data):
	for DOF in DOFs:
		for robot_num in range(robots_per_dof):
			if show_data:
				print("\nName: Manipulator_{}{}".format(DOF, robot_num))
				lengths, axes, dampings, frictions, gears = [],[],[],[],[]

			save_xml = make_model_xml("Manipulator_{}{}".format(DOF, robot_num))
			i = DOF-1	#Ignoring gripper joint/link
			
			for link_number in range(i):
				axis = randomize(Standard_axis)
				if link_number == 1:
					length = randomize(0.5)
				else:
					length = randomize(0.3)

				damping = randomize(Standard_damping)
				friction = randomize(Standard_friction)
				gear = randomize(Standard_joint_gear)
				save_xml.add_link(length, axis, joint_ranges[link_number], damping, friction, gear, link_number==i-1)

				if show_data:
					lengths.append([length])
					if link_number == 0:
						axis = [0, 0, 1]
					axes.append(axis)
					dampings.append([damping])
					frictions.append([friction])
					gears.append([gear])

			if show_data:
				print("lengths = {}".format(lengths))
				print("axes = {}".format(axes))
				print("dampings = {}".format(dampings))
				print("frictions = {}".format(frictions))
				print("gears = {}".format(gears))
			save_xml.save_file()

create_xmls(show_data)