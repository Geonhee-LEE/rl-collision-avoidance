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
from collections import deque

from model.net import QNetwork, CNNPolicy
from stage_world import StageWorld
from model.sac import sac_update_stage
from model.sac import generate_action
from model.update_file import hard_update
from model.replay_memory import ReplayMemory


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
#parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
#                    help='Automaically adjust α (default: False)')
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
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
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
args = parser.parse_args()


def run(comm, env, policy, critic, critic_opt, critic_target, policy_path, action_bound, optimizer):

    buff = []
    global_update = 0
    global_step = 0

    # world reset
    if env.index == 0: # step
        env.reset_world()

    update = 0 # The number of learning using Replay memory

    # replay_memory     
    replay_memory = ReplayMemory(args.replay_size, args.seed)

    for id in range(args.num_steps):
        
        # reset
        env.reset_pose()
        
        terminal = False
        ep_reward = 0
        step = 1

        # generate goal
        env.generate_goal_point()
        
        # get_state
        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]

        # episode 1
        while not terminal and not rospy.is_shutdown():
                        
            state_list = comm.gather(state, root=0)

            ## get_action
            #-------------------------------------------------------------------------
            # generate actions at rank==0
            a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)

            # execute actions
            #-------------------------------------------------------------------------            
            real_action = comm.scatter(scaled_action, root=0)
            
            ## run action
            #-------------------------------------------------------------------------            
            env.control_vel(real_action)

            rospy.sleep(0.001)

            ## get reward
            #-------------------------------------------------------------------------
            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1

            #-------------------------------------------------------------------------

            # get next state
            #-------------------------------------------------------------------------
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            #-------------------------------------------------------------------------

            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)
            state_next_list = comm.gather(state_next, root=0)


            ## training
            #-------------------------------------------------------------------------
            if env.index == 0: # Reset

                # add data in replay_memory
                #-------------------------------------------------------------------------
                replay_memory.push(state[0], state[1],state[2],a, logprob, r_list, state_next[0], state_next[1], state_next[2], terminal_list)
                if len(replay_memory) > args.batch_size:
            
                    ## update
                    #-------------------------------------------------------------------------
                    update = sac_update_stage(policy=policy, optimizer=optimizer, critic=critic, critic_opt=critic_opt, critic_target=critic_target, 
                                            batch_size=args.batch_size, memory=replay_memory, epoch=args.epoch, replay_size=args.replay_size,
                                            tau=args.tau, alpha=args.alpha, gamma=args.gamma, updates=update, update_interval=args.target_update_interval,
                                            num_step=args.batch_size, num_env=args.num_env, frames=args.laser_hist,
                                            obs_size=args.laser_beam, act_size=args.act_size)

                    buff = []
                    global_update += 1
                    update = update

            step += 1
            state = state_next

        ## save policy
        #--------------------------------------------------------------------------------------------------------------
        if env.index == 0: # Step is first
            if global_update != 0 and global_update % 1000 == 0:
                torch.save(policy.state_dict(), policy_path + '/policy_{}'.format(global_update))
                torch.save(critic.state_dict(), policy_path + '/critic_{}'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)


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

    env = StageWorld(beam_num=args.laser_beam, index=rank, num_env=args.num_env)
    print("Ready to environment")
    
    reward = None
    action_bound = [[0, -1], [1, 1]] #### Action maximum, minimum values
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'saved_policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=args.laser_hist, action_space=args.act_size)
        policy.cuda()
        
        opt = Adam(policy.parameters(), lr=args.lr)
        mse = nn.MSELoss()

        critic = QNetwork(frames=args.laser_hist, action_space=args.act_size)
        critic.cuda()

        critic_opt = Adam(critic.parameters(), lr=args.lr)
        critic_target = QNetwork(frames=args.laser_hist, action_space=args.act_size)
        critic_target.cuda()

        hard_update(critic_target, critic)

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage1_2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, critic=critic, critic_opt=critic_opt, critic_target=critic_target, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass
