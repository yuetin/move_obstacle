#!/usr/bin/env python3

import threading, queue
import time
import os
import shutil
import numpy as np
import math
import rospy
import tensorflow as tf
from sac_v15 import SAC
from env_v22 import Test
from manipulator_h_base_module_msgs.msg import P2PPose

MAX_EPISODES = 100000
MAX_EP_STEPS =  600
MEMORY_CAPACITY = 10000
BATTH_SIZE = 4
SIDE = ['right_', 'left_']
SIDE_ = ['R', 'L']
GOAL_REWARD = 800
LOAD = False
SAVE = [False, False]
COUNTER = [1, 1]
TRAIN_CNT = [0, 0]
EP = [0, 0]
WORKS = 1
SUCCESS_ARRAY = np.zeros([2,300])
GOAL_RATE = [10, 10]
ACTION_FLAG = [False, False]
ep_goal = [10,10]

def worker(name, workers, agent):
    global SUCCESS_ARRAY, ACTION_FLAG, SAVE, COUNTER, EP
    SUCCESS_RATE = 0
    COLLISION = False
    
    env = Test(name, workers) #0 = right
    time.sleep(0.5)
    print(threading.current_thread())
    print('name', name, 'workers', workers, 'agentID', id(agent))

    if RUN_FLAG[workers].is_set():
        RUN_FLAG[workers].clear()
    else:
        RUN_FLAG[workers].set()
    RUN_FLAG[workers].wait()
    t1 = time.time()
    while (not COORD.should_stop()) and (not rospy.is_shutdown()):
        if RUN_FLAG[workers].is_set():
            RUN_FLAG[workers].clear()
        else:
            RUN_FLAG[workers].set()
        RUN_FLAG[workers].wait()

        s_arr = []
        a_arr = []
        r_arr = []
        s__arr = []
        done_arr = []

        img_arr = []
        imgex_arr = []
        s, depth = env.reset()
        # print(s.shape)
        # print(depth.shape)
        
        ep_reward = 0
        success_cnt = 0
        done_cnt = 0
        EP[name] += 1
        ep = EP[name]
        SUCCESS_ARRAY[name, ep%300] = 0.
        # COLLISION = False
        first_fail = True
        for j in range(MAX_EP_STEPS):
            WORKER_EVENT[name].wait()
            a = agent.choose_action(s, depth)
            # print("cccccccccccccccccccccc")
            rd = np.random.rand()
            a *= (rd*3+0.5)
            
            s_, r, done, success, fail, succcccccccccccccckkkkk = env.step(a)
            # print("cccccccccccccccccccccc")
            # print(succcccccccccccccckkkkk.shape)
            # , succcccccccccccccckkkkk
            if j>10:
                s_arr.append(s)
                # print(len(s_arr))
                imgex_arr.append(depth)
                # list_shape = np.array(depth).shape
                # print(list_shape)
                a_arr.append(a)
                r_arr.append(r)
                s__arr.append(s_)
                img_arr.append(succcccccccccccccckkkkk)
                # list_shape = np.array(img_arr).shape
                # print(list_shape)
                done_arr.append(done)
                # agent.replay_buffer[workers].store_transition(s, a, r, s_, done)
                # if fail:
                #     if first_fail:
                #         first_fail = False
                #         for k in range(50):
                #             if k>=len(r_arr):
                #                 break
                #             r_arr[-k-1] -= (2-(k*0.04))
                #     else:
                #         r_arr[-1] -= 2

            success_cnt += int(success)
            done_cnt += int(done)
            # if collision:
            #     COLLISION = True
            s = s_
            ep_reward += r

            COUNTER[name]+=1
            if COUNTER[name] >= BATTH_SIZE*200 and COUNTER[name]%(80) == 0:
                WORKER_EVENT[name].clear()
                for _ in range(2+int(ep/1000)):
                    agent.learn(TRAIN_CNT[name])
                    TRAIN_CNT[name]+=1
                WORKER_EVENT[name].set()
                
                # LEARN_EVENT[name].set()
            if success_cnt > 10:
                # if not COLLISION:
                SUCCESS_ARRAY[name, ep%300] = 1.
                break
            # if done_cnt-success_cnt > 100:
            #     break
        
        for i in range(len(s_arr)):
            agent.replay_buffer[workers].store_transition(s_arr[i], imgex_arr[i], a_arr[i], r_arr[i], s__arr[i], img_arr[i], done_arr[i])
        s_arr.clear()
        a_arr.clear()
        r_arr.clear()
        s__arr.clear()
        done_arr.clear()
        img_arr.clear()
        imgex_arr.clear()

        SUCCESS_RATE = 0
        for z in SUCCESS_ARRAY[name]:
            SUCCESS_RATE += z/3
        if SUCCESS_RATE >= GOAL_RATE[name]:
        # if ep >= ep_goal[name]:
            SAVE[name] = True
            
        else:
            SAVE[name] = False
        agent.replay_buffer[workers].store_eprwd(ep_reward*j/100)
        
        if workers == 0 and SAVE[name]:
            SUCCESS_ARRAY[name] = np.zeros([300])
            save(agent, name)
            print('Running time: ', time.time() - t1)
        if env.is_success:
            print(SIDE_[name]+str(workers), ep, ' Reward: %i' % int(ep_reward), 'cnt: ',j, 's_rate: ', int(SUCCESS_RATE), 'sssuuucccccceeessssss', env.success_cnt)
        else:
            print(SIDE_[name]+str(workers), ep, ' Reward: %i' % int(ep_reward), 'cnt: ',j, 's_rate: ', int(SUCCESS_RATE), 'sssuuuccccckkkkmydick')

def save(agent, name):
    print(agent.path)
    if os.path.isdir(agent.path+str(GOAL_RATE[name])): shutil.rmtree(agent.path+str(GOAL_RATE[name]))
    os.mkdir(agent.path+str(GOAL_RATE[name]))
    ckpt_path = os.path.join(agent.path+str(GOAL_RATE[name]), 'SAC.ckpt')
    # if os.path.isdir(agent.path+str(ep_goal[name])): shutil.rmtree(agent.path+str(ep_goal[name]))
    # os.mkdir(agent.path+str(ep_goal[name]))
    # ckpt_path = os.path.join(agent.path+str(ep_goal[name]), 'SAC.ckpt')

    save_path = agent.saver.save(agent.sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)
    if GOAL_RATE[name] < 90:
        GOAL_RATE[name] += 5
    else:
        GOAL_RATE[name] += 2
    # if ep_goal[name] < 90:
    #     ep_goal[name] += 20
    # else:
    #     ep_goal[name] += 5
    # if GOAL_RATE[name] > 100:
    #     COORD.request_stop()

def train(name):
    global SAVE, COUNTER, RUN_FLAG
    threads_ = []
    print(threading.current_thread())
    env = Test(name, 0)
    agent = SAC(act_dim=env.act_dim, obs_dim=env.obs_dim, depth_dim=env.depth_dim,
            lr_actor=8e-3, lr_value=8e-3, gamma=0.99, tau=0.995, buffers = WORKS, name=SIDE[name], seed=name)
            # lr_actor=1e-3, lr_value=1e-3
    env = None
    print('name', name, 'agentID', id(agent))

    for j in range(WORKS):
        t = threading.Thread(target=worker, args=(name, j, agent,))
        threads_.append(t)
        if name == 0:
            RUN_FLAG.append(threading.Event())
            RUN_FLAG[j].set()
    time.sleep(1)
    for i in threads_:
        i.start()

if __name__ == '__main__':
    rospy.init_node('a')
    threads = []
    RUN_FLAG = []
    LEARN_EVENT = [threading.Event(), threading.Event()]
    WORKER_EVENT = [threading.Event(), threading.Event()]
    COORD = tf.train.Coordinator()

    for i in range(2):
        t = threading.Thread(target=train, args=(i,))
        threads.append(t)
        WORKER_EVENT[i].set()
        LEARN_EVENT[i].clear()

    for i in threads:
        i.start()
        time.sleep(10)
    print(threading.current_thread())
    COORD.join(threads)
