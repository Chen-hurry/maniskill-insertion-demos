# Planning Module / 规划模块

This directory contains motion-planning wrappers used to generate high-quality robot trajectories for robust recovery experiments.

本目录用于存放基于运动规划的数据采集代码，用来为鲁棒恢复实验生成高质量机械臂轨迹。

## panda_motion_planner.py

`panda_motion_planner.py` wraps the official ManiSkill Panda motion-planning solver instead of copying the full ManiSkill implementation.

`panda_motion_planner.py` 并不复制 ManiSkill 官方求解器，而是在本项目中封装和调用官方 Panda 运动规划器。

The official solver uses `mplib` with the robot URDF/SRDF model to plan joint-space trajectories toward target TCP poses.

官方求解器基于 `mplib`，利用机器人 URDF/SRDF 模型，把当前关节状态规划到目标末端执行器 TCP 位姿。

## Main Idea / 核心思路

The planner does not directly plan from object position to goal position. It plans robot motion from the current robot joint state to a target end-effector pose.

规划器不是直接规划“物体位置到目标位置”，而是规划“当前机器人关节状态到目标末端位姿”。

For pick-and-place, the task is decomposed into several waypoints:

对于 pick-and-place 任务，需要拆成多个阶段目标：

```text
current TCP
-> pre-grasp pose
-> grasp pose
-> close gripper
-> lift pose
-> pre-place pose
-> place pose
-> open gripper
