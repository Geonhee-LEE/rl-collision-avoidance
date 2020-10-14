import time
import rospy
import copy
import tf
import numpy as np
import random

from geometry_msgs.msg import Twist, Pose, Point32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8, Int32
#from model.utils import get_init_pose_24, get_goal_point_24

class StageWorld():
    def __init__(self, beam_num, index, num_env):
        self.index = index
        self.num_env = num_env

        #self.robot_radius = [robot_radius]

        node_name = 'StageEnv_' + str(index)
        rospy.init_node(node_name, anonymous=None)

        self.beam_mum = beam_num
        self.laser_cb_num = 0
        self.scan = None

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        
        self.goal_point = [0, 0]

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.
        # self.reset_pose = None

        #self.init_pose = get_init_pose_24(self.index)

        # for get reward and terminate
        self.stop_counter = 0

        # -----------Publisher and Subscriber-------------

        cmd_vel_topic = '/robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = '/robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)
        
        goal_point_topic = '/robot_' + str(index) + '/pub_goal_point'
        self.pub_goal_point = rospy.Publisher(goal_point_topic, Pose, queue_size=2)


        # ---------Subscriber-----------------

        object_state_topic = '/robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = '/robot_'+ str(index) + '/base_scan_0'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        dead_topic = '/robot_'+ str(index) + '/base_scan_1'
        self.dead_sub = rospy.Subscriber(dead_topic, LaserScan, self.laser_dead_callback)

        odom_topic = '/robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = '/robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)

        map_num_topic = '/env_num' 
        self.map_num = rospy.Subscriber(map_num_topic, Int32, self.map_env_callback)

        map_pub_topic = '/env_change' 
        self.map_pub = rospy.Publisher(map_pub_topic, Int32, queue_size=10)

        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # -----------Service-------------------
        
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)

        self.env_index = 0

        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        self.is_crashed = None


        self.scan_dead = None
        self.is_dead = None

        self.is_collision = 0
        self.lidar_danger = 1.0 #- robot_radius
        self.scan_min = 10.0

        while self.scan is None or self.speed is None or self.state is None or self.speed_GT is None or self.state_GT is None:
        
	    pass
       
        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)

    def map_change(self):
        map_index = random.randint(0, 5)
        print("rank: ", self.index, "map index: ", map_index)
        map_data = Int32()
        map_data.data = map_index
        self.map_pub.publish(map_data)

    def map_env_callback(self, map_num):
        self.env_index = map_num.data

    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1

    def laser_dead_callback(self, dead_scan):
        self.scan_dead = np.array(dead_scan.ranges)
        self.scan_dead[np.isnan(self.scan_dead)] = 0.5
        self.scan_dead[np.isinf(self.scan_dead)] = 0.5
        dead_scan_min = np.min(self.scan_dead)
        if dead_scan_min < 0.5:
            self.is_dead = True

    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):

        self.is_crashed = flag.data

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 10.0
        scan[np.isinf(scan)] = 10.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.

        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step

        sparse_scan_right = []
        index = raw_beam_num - 1.

        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step

        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 10.0 - 0.5

    def collision_laser_flag(self, r):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 10.0
        scan[np.isinf(scan)] = 10.0

        scan_min = np.min(scan)

        if scan_min <= r:
            self.is_collision = 1
        else:
            self.is_collision = 0

        self.scan_min = scan_min

    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        #rospy.sleep(0.5)
        rospy.sleep(0.5)
        
    def generate_goal_point(self):

        if self.env_index == 0:
            self.goal_point = self.generate_random_goal_1()
        elif self.env_index == 1:
            self.goal_point = self.generate_random_goal_2()
        elif self.env_index == 2:
            self.goal_point = self.generate_random_goal_3()
        elif self.env_index == 3:
            self.goal_point = self.generate_random_goal_4()
        elif self.env_index == 4:
            self.goal_point = self.generate_random_goal_5()
        elif self.env_index == 5:
            self.goal_point = self.generate_random_goal_6()
            
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)

    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()

        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()

        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)

        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        reward_t = 0
        reward_ct = 0
        result = 0

        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 25
            result = 'Reach Goal'
        
        if is_crash == 1:
            terminate = True
            reward_c = -25.
            result = 'Crashed'  
        '''
        if self.is_collision == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'
        '''
        if self.is_dead == 1:
            terminate = True
            reward_c = -25.
            result = 'Crashed'

        if np.abs(w) >  1.05:
            reward_w = -0.1 * np.abs(w)
        '''
        if (self.scan_min > self.robot_radius[0]) and (self.scan_min < (self.lidar_danger+self.robot_radius[0])):
            reward_ct = -0.25*((self.lidar_danger+self.robot_radius[0]) - self.scan_min)
        '''
        if t > 400:
            terminate = True
            result = 'Time out'
        else:
            reward_t = -0.05

        reward = reward_g + reward_c + reward_w + reward_ct + reward_t

        return reward, terminate, result


    def reset_pose(self):

        if self.env_index == 0:
            reset_pose = self.generate_random_pose_1()
        elif self.env_index == 1:
            reset_pose = self.generate_random_pose_2()
        elif self.env_index == 2:
            reset_pose = self.generate_random_pose_3()
        elif self.env_index == 3:
            reset_pose = self.generate_random_pose_4()
        elif self.env_index == 4:
            reset_pose = self.generate_random_pose_5()
        elif self.env_index == 5:
            reset_pose = self.generate_random_pose_6()
                
        rospy.sleep(1.0)

        self.control_pose(reset_pose)

        [x_robot, y_robot, theta] = self.get_self_stateGT()

        rospy.sleep(0.5)
        if not self.env_index == 3:
            while np.abs(reset_pose[0] - x_robot) > 0.2 or np.abs(reset_pose[1] - y_robot) > 0.2:
                [x_robot, y_robot, theta] = self.get_self_stateGT()
                self.control_pose(reset_pose)

                rospy.sleep(0.5)

        rospy.sleep(1.0)
        self.is_dead = False

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)


    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)

    #---------------------------------------------------------------------------

    def generate_random_pose_1(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        if self.index == 0:
            x = np.random.uniform(-12.5, -19)
            y = np.random.uniform(9, 10)

        elif self.index == 1:
            x = np.random.uniform(-6, -12.5)
            y = np.random.uniform(9, 10)

        elif self.index == 2:
            x = np.random.uniform(-12, -13.5)
            y = np.random.uniform(2, 10.5)

        else:
            x = np.random.uniform(-12, -13.5)
            y = np.random.uniform(10.5, 19)

        theta = np.random.uniform(0, 2*np.pi)

        return [x, y, theta]


    def generate_random_pose_2(self):

        [x_robot, y_robot, theta] = self.get_self_stateGT()

        if self.index == 0:
            x = np.random.uniform(-3, 1.5)
            y = np.random.uniform(14.5, 17)

        elif self.index == 1:
            x = np.random.uniform(-1.5, 5.5)
            y = np.random.uniform(14.5, 17)

        elif self.index == 2:
            x = np.random.uniform(-3, 1.5)
            y = np.random.uniform(2, 4)

        else:
            x = np.random.uniform(-1.5, 5.5)
            y = np.random.uniform(2, 4)

        theta = np.random.uniform(0, 2*np.pi)

        return [x, y, theta]

    def generate_random_pose_3(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        if self.index == 0:
                
            x = np.random.uniform(10, 14)
            y = np.random.uniform(2, 4)

        elif self.index == 1:
                
            x = np.random.uniform(10, 14)
            y = np.random.uniform(4, 6)

        elif self.index == 2:
                
            x = np.random.uniform(14, 18)
            y = np.random.uniform(2, 4)

        else:
            x = np.random.uniform(14, 18)
            y = np.random.uniform(4, 6)

        theta = np.random.uniform(0, 2*np.pi)

        return [x, y, theta]

    def generate_random_pose_4(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        x = np.random.uniform(-8, -18)
        y = np.random.uniform(-4, -18)

        dis_goal = np.sqrt((x + 13) ** 2 + (y + 10) ** 2)

        #while (dis_goal < 5) and not rospy.is_shutdown():
        #    x = np.random.uniform(-8, -18)
        #    y = np.random.uniform(-4, -18)
            
        
        theta = np.random.uniform(0, 2*np.pi)

        return [x, y, theta]

    def generate_random_pose_5(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        if self.index == 0:
                
            x = np.random.uniform(-3, 1)
            y = np.random.uniform(-16.5, -18.5)


        elif self.index == 1:
                
            x = np.random.uniform(1, 6)
            y = np.random.uniform(-16.5, -18.5)

        elif self.index == 2:
                
            x = np.random.uniform(-3, 1)
            y = np.random.uniform(-2, -5)

        else:
                
            x = np.random.uniform(1, 6)
            y = np.random.uniform(-2, -5)

        return [x, y, theta]


    def generate_random_pose_6(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        if self.index == 0:
                
            x = np.random.uniform(9, 11)
            y = np.random.uniform(-18, -13)


        elif self.index == 1:
                
            x = np.random.uniform(11.5, 13.5)
            y = np.random.uniform(-18, -13)

        elif self.index == 2:
                
            x = np.random.uniform(14, 16)
            y = np.random.uniform(-18, -13)

        else:
                
            x = np.random.uniform(16.5, 18.5)
            y = np.random.uniform(-18, -13)

        theta = np.random.uniform(0, 2*np.pi)

        return [x, y, theta]

    #--------------------------------------------------------------------------------

    def generate_random_goal_1(self):
        self.init_pose = self.get_self_stateGT()
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(-12, -13.5)
        y = np.random.uniform(2, 19)

        if self.index == 2 or self.index == 3:
            x = np.random.uniform(-6, -19)
            y = np.random.uniform(9, 10)

        else:
            x = np.random.uniform(-12, -13.5)
            y = np.random.uniform(2, 19)

        return [x, y]

    def generate_random_goal_2(self):
        self.init_pose = self.get_self_stateGT()
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        if self.index == 0 or self.index == 1:
            x = np.random.uniform(-3, 5.5)
            y = np.random.uniform(2, 4)

        else:            
            x = np.random.uniform(-3, 5.5)
            y = np.random.uniform(14.5, 17)

        return [x, y]


    def generate_random_goal_3(self):
        self.init_pose = self.get_self_stateGT()
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(10, 18)
        y = np.random.uniform(10, 14)

        return [x, y]

    def generate_random_goal_4(self):
        self.init_pose = self.get_self_stateGT()
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(-8, -18)
        y = np.random.uniform(-2, -18)

        #dis_origin = np.sqrt((x + 13) ** 2 + (y + 10) ** 2)
        dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)

        '''get_init_pose_24
        while dis_goal > 7 or dis_goal < 5 and not rospy.is_shutdown():
            x = np.random.uniform(-8, -18)
            y = np.random.uniform(-2, -18)

            #dis_origin = np.sqrt((x + 13) ** 2 + (y + 10) ** 2)
            dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)
        '''
        
        return [x, y]


    def generate_random_goal_5(self):
        self.init_pose = self.get_self_stateGT()
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        if self.index == 2 or self.index == 3:
                
            x = np.random.uniform(-3, 6)
            y = np.random.uniform(-16.5, -18.5)

        else:
                
            x = np.random.uniform(-3, 6)
            y = np.random.uniform(-2, -5)

        return [x, y]

    def generate_random_goal_6(self):
        self.init_pose = self.get_self_stateGT()
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 18)
        y = np.random.uniform(-3, -5)

        return [x, y]

    #----------------------------------------------------------------------------------------

    def generate_random_radius(self):
        self.robot_radius = round(np.random.uniform(0.3, 1.5), 3)
        

    
