import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
import argparse
from mpi4py import MPI

from torch.optim import Adam
import datetime
#from torch.utils.tensorboard import SummaryWriter
from collections import deque

from model.net import QNetwork, ValueNetwork, GaussianPolicy, DeterministicPolicy
from stage_world import StageWorld
from model.sac import SAC
from model.replay_memory import ReplayMemory


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Stage",
                    help='Environment name (default: Stage)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter \alpha determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust \alpha (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: True)') 
parser.add_argument('--laser_beam', type=int, default=512,
                    help='the number of Lidar scan [observation] (default: 512)')
parser.add_argument('--num_env', type=int, default=1,
                    help='the number of environment (default: 1)')
parser.add_argument('--laser_hist', type=int, default=3,
                    help='the number of laser history (default: 3)')
parser.add_argument('--act_size', type=int, default=2,
                    help='Action size (default: 2, translation, rotation velocity)')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')                    
args = parser.parse_args()


def run(comm, env, agent, args):

    # Training Loop
    total_numsteps = 0
    updates = 0

    # world reset
    if env.index == 0: # step
        env.reset_world()

    # replay_memory     
    replay_memory = ReplayMemory(args.replay_size, args.seed)

    #Tesnorboard
    #writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    #                                                         args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    for i_episode in range(args.num_steps):
        episode_reward = 0
        episode_steps = 0
        done = False        
        # Env reset
        env.reset_pose()
        # generate goal
        env.generate_goal_point()
        
        # Get initial state
        frame = env.get_laser_observation()
        frame_stack = deque([frame, frame, frame])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [frame_stack, goal, speed]

        # Episode start
        while not done and not rospy.is_shutdown():    
            state_list = comm.gather(state, root=0)

            ## Select action
            #----------------------------------------------
            if args.start_steps > total_numsteps:
                #action = env.action_space.sample()  # Sample random action
                action = agent.select_action(state_list)  # Sample action from policy
            else:
                action = agent.select_action(state_list)  # Sample action from policy

            print ("env.index: ", env.index, "select_action: ", action.shape)
            if len(replay_memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    #writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    #writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    #writer.add_scalar('loss/policy', policy_loss, updates)
                    #writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    #writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
                    
            # Execute actions
            #-------------------------------------------------------------------------            
            real_action = comm.scatter(action, root=0)  
            
            print ("env.index: ", env.index, "select_action2: ", real_action.shape)  
            env.control_vel(real_action)
            rospy.sleep(0.001)

            ## Get reward and terminal state
            reward, done, result = env.get_reward_and_terminate(episode_steps)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Get next state
            next_frame = env.get_laser_observation()
            left = frame_stack.popleft()
            
            frame_stack.append(next_frame)
            next_goal = np.asarray(env.get_local_goal())
            next_speed = np.asarray(env.get_self_speed())
            next_state = [frame_stack, next_goal, next_speed]

            r_list = comm.gather(reward, root=0)
            done_list = comm.gather(done, root=0)
            #next_state_list = comm.gather(next_state, root=0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == num_steps else float(not done)

            memory.push(frame_stack, goal, speed, action, reward, next_frame, next_goal, next_speed, mask) # Append transition to memory

            state = next_state  

        if total_numsteps > args.num_steps:
            break
        #writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))



if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD # There is one special communicator that exists when an MPI program starts, that contains all the processes in the MPI program. This communicator is called MPI.COMM_WORLD
    size = comm.Get_size() # The first of these is called Get_size(), and this returns the total number of processes contained in the communicator (the size of the communicator).
    rank = comm.Get_rank() # The second of these is called Get_rank(), and this returns the rank of the calling process within the communicator. Note that Get_rank() will return a different value for every process in the MPI program.
    print("MPI size=%d, rank=%d" % (size, rank))

    # Environment
    env = StageWorld(beam_num=args.laser_beam, index=rank, num_env=args.num_env)
    print("Ready to environment")
    
    reward = None
    if rank == 0:
        # Agent num_frame_obs, num_goal_obs, num_vel_obs, action_space, args
        action_bound = np.array([[0, -1], [1, 1]]) #### Action maximum, minimum values
        agent = SAC(env = env, num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)
    else:
        agent = None

    try:
        run(comm=comm, env=env, agent=agent, args=args)
    except KeyboardInterrupt:
        pass
