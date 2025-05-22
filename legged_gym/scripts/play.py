# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit,export_world_model, task_registry, Logger

import numpy as np
import torch
import pygame

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 32)
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.commands.heading_command = True

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    # env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]


    # env_cfg.terrain.mesh_type = 'plane'
    if(env_cfg.terrain.mesh_type == 'plane'):
        env_cfg.rewards.scales.feet_edge = 0
        env_cfg.rewards.scales.feet_stumble = 0


    if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt','discrete']):
        print('terrain should be one of slope, stair, gap, climb, crawl, and tilt, set to climb as default')
    args.terrain = 'discrete'
    
    env_cfg.terrain.terrain_proportions = {
        'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
        'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
        'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
        'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
        'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
        'discrete'  : [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
     }[args.terrain]
    
    env_cfg.commands.ranges.lin_vel_x = [0,0]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]

    env_cfg.commands.ranges.flat_lin_vel_x = [0,0]
    env_cfg.commands.ranges.flat_lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

    env_cfg.depth.use_camera = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'WMP'


    train_cfg.runner.checkpoint = -1
    ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)


    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    history_length = 5
    trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
                                            env.privileged_dim - env.height_dim -env_cfg.env.forward_height_dim - 3), device = env.device)
    obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                        obs[:, env.privileged_dim + 9:-(env.height_dim+env_cfg.env.forward_height_dim)]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    world_model = ppo_runner._world_model.to(env.device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions),
                                    device=env.device)
    wm_obs = {
        "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
        "is_first": wm_is_first,
    }

    if (env.cfg.depth.use_camera):
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                      device=world_model.device)

    path = os.path.join(path,'world_model')
    
    # export_world_model(ppo_runner._world_model, path,wm_obs)


    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)



    command = torch.tensor([0,0,0], device=env.device)
    # 初始化 pygame
    pygame.init()
    if pygame.joystick.get_count() == 0:
        print("未检测到手柄，请连接手柄后重试！")
        exit()
    # 初始化手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"已连接手柄：{joystick.get_name()}")


    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)
    while True:  # 添加无限循环
        # 主循环：执行单个episode的多个时间步（与环境最大步长相关）
        for i in range(1000*int(env.max_episode_length) + 3):
            '''
            # 世界模型更新条件：当全局计数器达到更新间隔的整数倍时
            if (env.global_counter % wm_update_interval == 0):
                
                # 如果启用深度相机，更新观测中的图像数据
                if (env.cfg.depth.use_camera):
                    # 从环境信息中获取深度图，增加一个维度后传输到世界模型设备
                    wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)

                    wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                      device=world_model.device)
                    
                # 世界模型编码：将观测数据编码为潜在表示
                wm_embed = world_model.encoder(wm_obs)
                
                # 动态模型观测步进：根据当前潜在状态、动作和观测更新潜在状态
                wm_latent, _ = world_model.dynamics.obs_step(
                    wm_latent,          # 当前潜在状态
                    wm_action,          # 历史动作序列
                    wm_embed,           # 编码后的观测
                    wm_obs["is_first"], # 是否是初始状态的标记
                    sample=True         # 启用随机采样
                )
                
                # 提取确定性特征：用于策略网络的输入
                wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
                
                # 重置首次标记：完成首次更新后设为False
                wm_is_first[:] = 0
            '''
            # 轨迹历史处理：将多维历史数据展平为二维（批处理维度 x 时间步特征）
            history = trajectory_history.flatten(1).to(env.device)
            
            # 策略推理：使用当前观测、历史轨迹和世界模型特征生成动作
            # detach()切断梯度反向传播（推理阶段不需要梯度）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return           
            x = -joystick.get_axis(1)
            y = joystick.get_axis(0) 
            z = -joystick.get_axis(3)
            
            command = torch.tensor([x,y,z], device=env.device)
            if args.terrain == 'gap' or args.terrain == 'climb' or args.terrain == 'tilt':
                command = torch.tensor([0.6,0,0], device=env.device)
            #print(command)
            obs[:, 55 + 6:55 + 9] = command
            actions = policy(
                obs.detach(),        # 当前环境观测
                history.detach(),    # 历史轨迹数据
                wm_feature.detach()  # 世界模型特征
            )

            # 环境步进：执行动作获取新状态和奖励
            # _ 表示中间变量（通常为terminated状态标记）
            # reset_env_ids 是需要重置的环境ID列表
            obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

            # 累计未终止环境的奖励：
            # 使用逐元素乘法过滤已终止环境的奖励
            not_dones *= (~dones)
            total_reward += torch.mean(rews * not_dones)

            # 更新世界模型的动作历史记录：
            # 滑动窗口机制，保留最近N个动作
            wm_action_history = torch.concat(
                (wm_action_history[:, 1:],   # 保留除最早外的所有历史动作
                actions.unsqueeze(1)),      # 添加新动作并增加时间步维度
                dim=1
            )
            
            # 构建新的世界模型观测：
            wm_obs = {
                "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],  # 本体感知特征
                "is_first": wm_is_first,  # 重置标记
            }
            
            # 如果使用相机，初始化图像观测缓冲区
            if (env.cfg.depth.use_camera):
                wm_obs["image"] = torch.zeros(
                    ((env.num_envs,) + env.cfg.depth.resized + (1,)),
                    device=world_model.device
                )

            # 处理需要重置的环境：
            reset_env_ids = reset_env_ids.cpu().numpy()
            if (len(reset_env_ids) > 0):
                wm_action_history[reset_env_ids, :] = 0  # 重置动作历史
                wm_is_first[reset_env_ids] = 1           # 标记需要重新初始化

            # 准备世界模型需要的动作输入格式（展平时间步维度）
            wm_action = wm_action_history.flatten(1)

            # 轨迹历史维护：
            env_ids = dones.nonzero(as_tuple=False).flatten()
            trajectory_history[env_ids] = 0  # 重置已终止环境的轨迹
            
            # 构建无命令的观测特征（用于历史记录）：
            obs_without_command = torch.concat(
                (obs[:, env.privileged_dim:env.privileged_dim + 6],        # 本体状态
                obs[:, env.privileged_dim + 9:-(env.height_dim+env_cfg.env.forward_height_dim)]),          # 传感器数据
                dim=1
            )
            
            # 更新轨迹历史：滑动窗口机制，保留最近N个观测
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:],            # 保留除最早外的历史
                obs_without_command.unsqueeze(1)),    # 添加新观测并增加时间步维度
                dim=1
            )

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                lootat = env.root_states[8, :3]
                camara_position = lootat.detach().cpu().numpy() + [0, 1, 0]
                env.set_camera(camara_position, lootat)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()

        print('total reward:', total_reward)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
