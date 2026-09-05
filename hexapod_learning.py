import os
import shutil
from datetime import datetime
import numpy as np
import itertools
import torch
import torch.nn as nn
import time
from typing import Callable, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sqlite3
import glob
import random
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# =====================================================================
# 1. PARAMETRY MODELU FIZYCZNEGO I ŚRODOWISKA
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "model", "urdf", "model.urdf")
TARGET_HEIGHT = 0.228      
FOOT_INDICES = [2, 5, 8, 11, 14, 17]
KNEE_INDICES = [2, 5, 8, 11, 14, 17]
MAX_TARGET_VELOCITY = 0.3      

SPRING_COEFFICIENT = 0.35       
DAMPING_COEFFICIENT = 0.9     
SMOOTHING_COEFFICIENT = 1.0   
LIGHT_CONTACT_THRESHOLD = 1.0
FULL_CONTACT_THRESHOLD = 4.0

WEIGHT_HEIGHT = -150.0            
WEIGHT_VELOCITY = 1000.0           
WEIGHT_ROLL_PITCH = -30.0
WEIGHT_MACRO_DRIFT = -400.0
WEIGHT_YAW = -800.0
WEIGHT_YAW_SOFT = -5.0
REWARD_SCALING = 2000.0         
WEIGHT_FOOT_DRAG = -800.0      
MIN_SWING_HEIGHT = 0.07        
WEIGHT_SELF_COLLISION = -500.0   

FALL_LIMIT_RAD = 1.57         
MIN_BELLY_HEIGHT = 0.105
UNDERGROUND_LIMIT = -0.5
MAX_EPISODE_STEPS = 1000  
WEIGHT_OVEREXTENSION = -50.0     
MIN_KNEE_ANGLE = 0.2             

POLICY_FREQUENCY = 50
PHYSICS_FREQUENCY = 250
FRAME_SKIP = PHYSICS_FREQUENCY // POLICY_FREQUENCY

REF_STRIDE_LENGTH = 0.15
MIN_FREQUENCY = 0.0
MAX_FREQUENCY = 3.0

MAX_SERVO_DEVIATION_RAD = 0.8028 
ACTION_SCALING_VECTOR = np.full(18, MAX_SERVO_DEVIATION_RAD, dtype=np.float32)

policy_architecture = dict(
    activation_fn=nn.ReLU, 
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
)

# =====================================================================
# ŚRODOWISKO UCZENIA ZE WZMOCNIENIEM
# =====================================================================
class HexapodRLTrainingEnv(gym.Env):
    def __init__(self, render_mode=None, min_test_velocity=None, target_grid=None, enable_noise=False):
        super().__init__()
        self.render_mode = render_mode
        self.min_test_velocity = min_test_velocity
        self.enable_noise = enable_noise
        self.noise_probability = 0.5
        self.target_grid = target_grid
        self.current_target_index = 0 

        self.global_path_start = np.array([0.0, 0.0])
        self.global_direction_vector = np.array([1.0, 0.0])

        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_path = URDF_PATH

        self.num_joints = 18
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.history_k = 1  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)

        self.action_history_buffer = np.zeros((self.history_k, self.num_joints), dtype=np.float32)
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0

        self.default_pose = np.array([
            0.0,  1.047198, 0.523599,  
            0.0,  1.047198, 0.523599,  
            0.0,  1.047198, 0.523599,  
            0.0, -1.047198, -0.523599, 
            0.0, -1.047198, -0.523599, 
            0.0, -1.047198, -0.523599  
        ])

        self.step_counter = 0
        self.simulation_time = 0.0
        self.gait_phases = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi], dtype=np.float32)
        self.total_lateral_drift = 0.0
        self.penalty_multiplier = 0.0  
        p.setTimeStep(1.0 / PHYSICS_FREQUENCY)
        self.rotation_drift_multiplier = 1.0

    def _generate_target_velocity(self):
        if self.target_grid is not None:
            angle, v = self.target_grid[self.current_target_index]
            self.current_target_index = (self.current_target_index + 1) % len(self.target_grid)
            vx = v * np.cos(angle)
            vy = v * np.sin(angle)
            return vx, vy
        else:
            lower_limit = 0.05 if self.min_test_velocity is None else self.min_test_velocity
            chance = np.random.rand()
            if chance < 0.60:
                direction = np.random.choice([0, 1, 2, 3])
                v = np.random.uniform(lower_limit, MAX_TARGET_VELOCITY)
                if direction == 0:   return v, 0.0
                elif direction == 1: return -v, 0.0
                elif direction == 2: return 0.0, v
                else:               return 0.0, -v
            else:
                v = np.random.uniform(lower_limit, MAX_TARGET_VELOCITY)
                angle = np.random.uniform(0, 2 * np.pi)
                return v * np.cos(angle), v * np.sin(angle)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.physicsClient)
        
        self.global_path_start = np.array([0.0, 0.0])
        self.global_direction_vector = np.array([1.0, 0.0])

        p.changeDynamics(self.planeId, -1, lateralFriction=1.0, restitution=0.0, physicsClientId=self.physicsClient)

        start_pos = [0, 0, TARGET_HEIGHT]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robotId = p.loadURDF(self.urdf_path, start_pos, start_orientation, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.physicsClient)

        for j in range(-1, p.getNumJoints(self.robotId, physicsClientId=self.physicsClient)):
            p.changeDynamics(self.robotId, j, restitution=0.0, lateralFriction=1.0, contactStiffness=1e6, contactDamping=1e5, physicsClientId=self.physicsClient)

        self.joint_indices = []
        self.joint_max_forces = []
        self.joint_max_velocities = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []

        for j in range(p.getNumJoints(self.robotId, physicsClientId=self.physicsClient)):
            info = p.getJointInfo(self.robotId, j, physicsClientId=self.physicsClient)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(j)
                self.joint_max_forces.append(info[10])
                self.joint_max_velocities.append(info[11])
                self.joint_lower_limits.append(info[8])
                self.joint_upper_limits.append(info[9])

        self.joint_lower_limits = np.array(self.joint_lower_limits, dtype=np.float32)
        self.joint_upper_limits = np.array(self.joint_upper_limits, dtype=np.float32)

        for i, j in enumerate(self.joint_indices):
            p.resetJointState(self.robotId, j, targetValue=self.default_pose[i], physicsClientId=self.physicsClient)
            p.setJointMotorControl2(
                bodyUniqueId=self.robotId, jointIndex=j, controlMode=p.POSITION_CONTROL,
                targetPosition=self.default_pose[i], force=self.joint_max_forces[i],
                positionGain=SPRING_COEFFICIENT, velocityGain=DAMPING_COEFFICIENT,
                maxVelocity=self.joint_max_velocities[i], physicsClientId=self.physicsClient
            )

        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physicsClient)

        self.action_history_buffer = np.zeros((self.history_k, self.num_joints), dtype=np.float32)
        self.step_counter = 0
        self.simulation_time = 0.0
        self.gait_phases = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi], dtype=np.float32)
        self.total_lateral_drift = 0.0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        
        self.target_cmd_vx, self.target_cmd_vy = self._generate_target_velocity()
        self.steps_until_target_change = np.random.randint(200, 400)

        base_pos, base_ori = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.physicsClient)
        euler = p.getEulerFromQuaternion(base_ori)
        self.target_yaw = euler[2] 

        self.global_path_start = np.array([base_pos[0], base_pos[1]])
        local_x = self.target_cmd_vy
        local_y = -self.target_cmd_vx

        vx_glob = local_x * np.cos(self.target_yaw) - local_y * np.sin(self.target_yaw)
        vy_glob = local_x * np.sin(self.target_yaw) + local_y * np.cos(self.target_yaw)
        
        length = np.sqrt(vx_glob**2 + vy_glob**2)
        if length > 0.001:
            self.global_direction_vector = np.array([vx_glob, vy_glob]) / length
        else:
            self.global_direction_vector = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw)])

        return self._get_obs(), {}

    def _update_phase_clock(self, dt):
        velocity_module = np.sqrt(self.cmd_vx**2 + self.cmd_vy**2)
        if velocity_module >= 0.01:
            frequency = np.clip(velocity_module / REF_STRIDE_LENGTH, MIN_FREQUENCY, MAX_FREQUENCY)
            omega = 2.0 * np.pi * frequency
            self.gait_phases += omega * dt

    def set_rotation_drift_multiplier(self, value):
        self.rotation_drift_multiplier = value

    def step(self, action):
        self.steps_until_target_change -= 1
        if self.steps_until_target_change <= 0:
            self.target_cmd_vx, self.target_cmd_vy = self._generate_target_velocity()
            self.steps_until_target_change = np.random.randint(200, 400)

            base_pos, base_ori = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.physicsClient)
            euler = p.getEulerFromQuaternion(base_ori)
            yaw = euler[2] 

            self.global_path_start = np.array([base_pos[0], base_pos[1]])
            local_x = self.target_cmd_vy
            local_y = -self.target_cmd_vx

            vx_glob = local_x * np.cos(yaw) - local_y * np.sin(yaw)
            vy_glob = local_x * np.sin(yaw) + local_y * np.cos(yaw)
            
            length = np.sqrt(vx_glob**2 + vy_glob**2)
            if length > 0.001:
                self.global_direction_vector = np.array([vx_glob, vy_glob]) / length
            else:
                self.global_direction_vector = np.array([np.cos(yaw), np.sin(yaw)])

            self.total_lateral_drift = 0.0 

        if hasattr(self, 'target_cmd_vx'):
            self.cmd_vx = 0.90 * self.cmd_vx + 0.10 * self.target_cmd_vx
            self.cmd_vy = 0.90 * self.cmd_vy + 0.10 * self.target_cmd_vy

        dt = 1.0 / POLICY_FREQUENCY
        self.simulation_time += dt

        for idx, foot_link_id in enumerate(FOOT_INDICES):
            contacts = p.getContactPoints(bodyA=self.robotId, bodyB=self.planeId, linkIndexA=foot_link_id, physicsClientId=self.physicsClient)
            
            touched_ground = False
            if len(contacts) > 0:
                max_force = max([k[9] for k in contacts])
                if max_force >= LIGHT_CONTACT_THRESHOLD:
                    touched_ground = True

            phase_mod = self.gait_phases[idx] % (2.0 * np.pi)

            if touched_ground and (np.pi / 2.0 < phase_mod < np.pi):
                self.gait_phases[idx] = np.pi

        self._update_phase_clock(dt)

        last_action = self.action_history_buffer[-1, :]
        smoothed_action = ((1.0 - SMOOTHING_COEFFICIENT) * last_action + SMOOTHING_COEFFICIENT * action)
        
        target_angles = self.default_pose + (smoothed_action * ACTION_SCALING_VECTOR)
        target_angles = np.clip(target_angles, self.joint_lower_limits, self.joint_upper_limits)

        for _ in range(FRAME_SKIP):
            for i, j in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId, jointIndex=j, controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angles[i], force=self.joint_max_forces[i],
                    positionGain=SPRING_COEFFICIENT, velocityGain=DAMPING_COEFFICIENT,
                    maxVelocity=self.joint_max_velocities[i], physicsClientId=self.physicsClient
                )
            p.stepSimulation(physicsClientId=self.physicsClient)

        obs = self._get_obs()
        self.step_counter += 1

        base_com_pos, base_com_ori = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.physicsClient)
        linear_vel_world_raw, angular_vel_world_raw = p.getBaseVelocity(self.robotId, physicsClientId=self.physicsClient) 

        dynamics_info = p.getDynamicsInfo(self.robotId, -1, physicsClientId=self.physicsClient)
        inertial_local_pos, inertial_local_ori = dynamics_info[3], dynamics_info[4]
        inv_inertial_pos, inv_inertial_ori = p.invertTransform(inertial_local_pos, inertial_local_ori)
        urdf_base_pos, urdf_base_ori = p.multiplyTransforms(base_com_pos, base_com_ori, inv_inertial_pos, inv_inertial_ori)

        euler_imu = p.getEulerFromQuaternion(urdf_base_ori)

        v_lin_global, angular_vel_world_raw = p.getBaseVelocity(self.robotId, physicsClientId=self.physicsClient) 
        _, inv_urdf_ori = p.invertTransform([0, 0, 0], urdf_base_ori)
        local_vel, _ = p.multiplyTransforms([0, 0, 0], inv_urdf_ori, v_lin_global, [0, 0, 0, 1])
        local_angular_vel, _ = p.multiplyTransforms([0, 0, 0], inv_urdf_ori, angular_vel_world_raw, [0, 0, 0, 1])

        actual_vx = -local_vel[1]  
        actual_vy = local_vel[0]  

        err_vx = actual_vx - self.cmd_vx
        err_vy = actual_vy - self.cmd_vy
        total_vel_error = np.sqrt(err_vx**2 + err_vy**2)

        target_module = np.sqrt(self.cmd_vx**2 + self.cmd_vy**2)

        if target_module < 0.005:
            relative_error = 0.0  
        else:
            relative_error = total_vel_error / target_module

        DELTA = 0.01
        smooth_error = np.sqrt(relative_error**2 + DELTA**2) - DELTA
        
        DROP_RATE = 1100.0 
        raw_velocity_reward = WEIGHT_VELOCITY - (DROP_RATE * smooth_error)
        
        BETA = 0.01
        direction_reward = np.logaddexp(0.0, BETA * raw_velocity_reward) / BETA

        current_xy = np.array([base_com_pos[0], base_com_pos[1]])
        traveled_vector = current_xy - self.global_path_start
        distance_along_path = np.dot(traveled_vector, self.global_direction_vector)
        point_on_path = self.global_path_start + distance_along_path * self.global_direction_vector
        self.total_lateral_drift = np.linalg.norm(current_xy - point_on_path)

        if abs(self.target_cmd_vx) < 0.01 and abs(self.target_cmd_vy) < 0.01:
            macro_drift_penalty = 0.0  
        else:
            DRIFT_DELTA = 0.005
            forgiven_drift = np.sqrt(self.total_lateral_drift**2 + DRIFT_DELTA**2) - DRIFT_DELTA
            macro_drift_penalty = (WEIGHT_MACRO_DRIFT * forgiven_drift) * self.rotation_drift_multiplier

        yaw_error = euler_imu[2] - getattr(self, 'target_yaw', 0.0)
        orientation_penalty = (WEIGHT_YAW * (1.0 - np.cos(yaw_error)) + WEIGHT_YAW_SOFT * abs(local_angular_vel[2])) * self.rotation_drift_multiplier

        HEIGHT_TOLERANCE = 0.00 
        height_err = abs(urdf_base_pos[2] - TARGET_HEIGHT)
        height_err_clamped = max(0.0, height_err - HEIGHT_TOLERANCE)
        HEIGHT_DELTA = 0.005
        smooth_height_err = np.sqrt(height_err_clamped**2 + HEIGHT_DELTA**2) - HEIGHT_DELTA
        height_penalty = WEIGHT_HEIGHT * smooth_height_err

        pitch_roll_penalty = WEIGHT_ROLL_PITCH * (euler_imu[0]**2 + euler_imu[1]**2)

        actual_module = np.sqrt(actual_vx**2 + actual_vy**2)
        paralysis_penalty = 0.0
        
        if target_module >= 0.005 and actual_module < 0.01:
            paralysis_penalty = -20000.0  
        
        foot_drag_penalty = 0.0
        for idx, foot_link_id in enumerate(FOOT_INDICES):
            phase_mod = self.gait_phases[idx] % (2.0 * np.pi)
            if 0.0 < phase_mod < np.pi:
                aabb = p.getAABB(self.robotId, foot_link_id, physicsClientId=self.physicsClient)
                z_min = aabb[0][2]
                if z_min < MIN_SWING_HEIGHT:
                    missing_height = MIN_SWING_HEIGHT - z_min
                    foot_drag_penalty += (missing_height ** 2)
        foot_drag_penalty *= WEIGHT_FOOT_DRAG

        overextension_penalty = 0.0
        knee_states = p.getJointStates(self.robotId, KNEE_INDICES, physicsClientId=self.physicsClient)
        for state in knee_states:
            abs_angle = abs(state[0])
            if abs_angle < MIN_KNEE_ANGLE:
                missing_flex = MIN_KNEE_ANGLE - abs_angle
                overextension_penalty += (missing_flex ** 2)
        overextension_penalty *= WEIGHT_OVEREXTENSION

        self_collision_penalty = 0.0
        self_contacts = p.getContactPoints(bodyA=self.robotId, bodyB=self.robotId, physicsClientId=self.physicsClient)
        for contact in self_contacts:
            distance = contact[8]
            base_pen = 1.0
            depth_pen = 0.0
            if distance < 0.0:
                depth_pen = abs(distance) * 100.0
            self_collision_penalty += (base_pen + depth_pen)
        self_collision_penalty *= WEIGHT_SELF_COLLISION

        # Kalkulacja finalnej nagrody MDP
        reward = (height_penalty + pitch_roll_penalty +
                  direction_reward + orientation_penalty +
                  macro_drift_penalty + paralysis_penalty +
                  foot_drag_penalty + overextension_penalty + self_collision_penalty) / REWARD_SCALING

        fall_side = abs(euler_imu[0]) > FALL_LIMIT_RAD
        fall_front = abs(euler_imu[1]) > FALL_LIMIT_RAD
        under_texture = bool(urdf_base_pos[2] < UNDERGROUND_LIMIT)
        too_low = bool(urdf_base_pos[2] < MIN_BELLY_HEIGHT)

        terminated = bool(fall_side or fall_front or under_texture or too_low)
        truncated = bool(self.step_counter >= MAX_EPISODE_STEPS)

        if terminated:
            reward -= 5.0 

        self.action_history_buffer = np.roll(self.action_history_buffer, shift=-1, axis=0)
        self.action_history_buffer[-1, :] = action.copy()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        _, base_ori = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.physicsClient)
        euler_imu = list(p.getEulerFromQuaternion(base_ori))

        _, angular_vel_world_raw = p.getBaseVelocity(self.robotId, physicsClientId=self.physicsClient)
        _, inv_urdf_ori = p.invertTransform([0, 0, 0], base_ori)
        local_angular_vel, _ = p.multiplyTransforms([0, 0, 0], inv_urdf_ori, angular_vel_world_raw, [0, 0, 0, 1])
        local_angular_vel = list(local_angular_vel)

        if self.enable_noise:
            euler_imu[0] += np.random.normal(0.0, 0.03) 
            euler_imu[1] += np.random.normal(0.0, 0.03)
            euler_imu[2] += np.random.normal(0.0, 0.08)
            local_angular_vel[2] += np.random.normal(0.0, 0.1)

        foot_contacts = np.zeros(12, dtype=np.float32)
        for idx, foot_link_id in enumerate(FOOT_INDICES):
            contacts = p.getContactPoints(bodyA=self.robotId, bodyB=self.planeId, linkIndexA=foot_link_id, physicsClientId=self.physicsClient)
            if len(contacts) > 0:
                max_force = max([k[9] for k in contacts])
                if max_force >= LIGHT_CONTACT_THRESHOLD:
                    sensor_amnesia = False
                    if self.enable_noise and np.random.rand() < self.noise_probability:
                        sensor_amnesia = True  
                    
                    if not sensor_amnesia:
                        foot_contacts[idx * 2] = 1.0 
                        if max_force >= FULL_CONTACT_THRESHOLD:
                            foot_contacts[idx * 2 + 1] = 1.0 

        velocity_cmd_input = np.array([self.cmd_vy / MAX_TARGET_VELOCITY, self.cmd_vx / MAX_TARGET_VELOCITY], dtype=np.float32)

        yaw_err = euler_imu[2] - getattr(self, 'target_yaw', euler_imu[2])
        yaw_sin = np.sin(yaw_err)
        yaw_cos = np.cos(yaw_err)

        phases_sin = np.sin(self.gait_phases)
        phases_cos = np.cos(self.gait_phases)
        normalized_phases = np.concatenate([phases_sin, phases_cos]).astype(np.float32)

        return np.concatenate([
            euler_imu[0:2],           
            [yaw_sin, yaw_cos],       
            [local_angular_vel[2]],   
            foot_contacts,               
            velocity_cmd_input,                
            normalized_phases,      
            self.action_history_buffer.flatten() 
        ]).astype(np.float32)

    def close(self):
        p.disconnect(physicsClientId=self.physicsClient)


def evaluate_model_performance(model_path, local_storage):
    MAGIC_SEED = 42
    np.random.seed(MAGIC_SEED)
    random.seed(MAGIC_SEED)
    torch.manual_seed(MAGIC_SEED)
    torch.use_deterministic_algorithms(True) 
    
    network_name = os.path.basename(model_path).replace('.zip', '')
    db_path = os.path.join(local_storage, "evaluation_metrics.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            network_name TEXT PRIMARY KEY,
            mean_target_velocity REAL,
            mean_yaw_error REAL,
            max_yaw_difference REAL,
            mean_lateral_drift REAL,
            drift_integral REAL,
            overall_score REAL,
            model_path TEXT,
            telemetry_plot_path TEXT,
            trajectory_plot_path TEXT
        )
    """)
    conn.commit()

    cursor.execute("SELECT 1 FROM evaluation_results WHERE network_name = ?", (network_name,))
    if cursor.fetchone():
        conn.close()
        return

    print(f"\n[EVALUATION] Rozpoczęto test referencyjny dla modelu: {network_name}...")
    
    try:
        eval_model = PPO.load(model_path, device="cpu") 
    except Exception as e:
        print(f"[EVALUATION] Błąd ładowania: {e}")
        conn.close()
        return

    eval_env = HexapodRLTrainingEnv(render_mode="direct")
    RECORDING_TIME = 16.0 
    dt_eval = 1.0 / POLICY_FREQUENCY

    def step_generator(t):
        if 0 <= t < 4: return 0.2, 0.0
        elif 4 <= t < 8: return 0.0, 0.2
        elif 8 <= t < 12: return -0.2, 0.0
        else: return 0.0, -0.2

    obs, _ = eval_env.reset(seed=42)
    current_time = 0.0
    telemetry_data = {'Time_s': [], 'Cmd_VX': [], 'Cmd_VY': [], 'Pos_X': [], 'Pos_Y': [], 'Pos_Z': [], 
                'Roll_deg': [], 'Pitch_deg': [], 'Yaw_deg': [], 'V_lin_X_Local': [], 
                'V_lin_Y_Local': [], 'Lateral_Drift': []}

    physics_id = eval_env.unwrapped.physicsClient
    start_pos, _ = p.getBasePositionAndOrientation(eval_env.unwrapped.robotId, physicsClientId=physics_id)
    start_x, start_y = start_pos[0], start_pos[1]
    side_len = 0.8

    while current_time <= RECORDING_TIME:
        idx_vx, idx_vy = step_generator(current_time)
        
        eval_env.unwrapped.target_cmd_vx = idx_vx
        eval_env.unwrapped.target_cmd_vy = idx_vy
        eval_env.unwrapped.steps_until_target_change = 999999 

        action, _ = eval_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        
        robot_id = eval_env.unwrapped.robotId
        
        pos, ori = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(ori)
        
        v_lin_global, _ = p.getBaseVelocity(robot_id, physicsClientId=physics_id)
        _, inv_urdf_ori = p.invertTransform([0, 0, 0], ori)
        v_lin_local, _ = p.multiplyTransforms([0, 0, 0], inv_urdf_ori, v_lin_global, [0, 0, 0, 1])

        actual_absolute_drift = 0.0
        phase_time = 4.0
        
        if 0 <= current_time < phase_time:
            actual_absolute_drift = abs(pos[0] - start_x)
        elif phase_time <= current_time < 2 * phase_time:
            actual_absolute_drift = abs(pos[1] - (start_y - side_len))
        elif 2 * phase_time <= current_time < 3 * phase_time:
            actual_absolute_drift = abs(pos[0] - (start_x + side_len))
        elif 3 * phase_time <= current_time <= RECORDING_TIME + 0.1:
            actual_absolute_drift = abs(pos[1] - start_y)
            
        eval_env.unwrapped.total_lateral_drift = actual_absolute_drift

        telemetry_data['Time_s'].append(current_time)
        telemetry_data['Cmd_VX'].append(eval_env.unwrapped.cmd_vx)
        telemetry_data['Cmd_VY'].append(eval_env.unwrapped.cmd_vy)
        telemetry_data['Pos_X'].append(pos[0]); telemetry_data['Pos_Y'].append(pos[1]); telemetry_data['Pos_Z'].append(pos[2])
        telemetry_data['Roll_deg'].append(np.degrees(roll)); telemetry_data['Pitch_deg'].append(np.degrees(pitch)); telemetry_data['Yaw_deg'].append(np.degrees(yaw))
        telemetry_data['V_lin_X_Local'].append(-v_lin_local[1]); telemetry_data['V_lin_Y_Local'].append(v_lin_local[0])
        telemetry_data['Lateral_Drift'].append(actual_absolute_drift)

        current_time += dt_eval
        if terminated or truncated:
            obs, _ = eval_env.reset(seed=42)
            start_pos, _ = p.getBasePositionAndOrientation(eval_env.unwrapped.robotId, physicsClientId=physics_id)
            start_x, start_y = start_pos[0], start_pos[1]

    eval_env.close()
    
    df = pd.DataFrame(telemetry_data)
    df['Cmd_Mod'] = np.sqrt(df['Cmd_VX']**2 + df['Cmd_VY']**2)
    df['Compliant_Vel'] = np.where(df['Cmd_Mod'] > 0.01, (df['Cmd_VX'] * df['V_lin_X_Local'] + df['Cmd_VY'] * df['V_lin_Y_Local']) / df['Cmd_Mod'], 0.0)
    
    mean_compliant_v = float(df[df['Cmd_Mod'] > 0.01]['Compliant_Vel'].mean())
    mean_yaw_err = float(df['Yaw_deg'].abs().mean())
    max_yaw = float(df['Yaw_deg'].max() - df['Yaw_deg'].min())
    mean_drift = float(df['Lateral_Drift'].mean())
    drift_integral = float(df['Lateral_Drift'].sum() * dt_eval)
    final_score = (mean_compliant_v * 1000.0) - (drift_integral * 150.0) - (mean_yaw_err * 15.0) - (max_yaw * 5.0)

    fig_traj, ax_traj = plt.subplots(figsize=(8, 8))
    ax_traj.plot(df['Pos_X'], df['Pos_Y'], color='darkviolet', linewidth=2.5, label='Trajektoria')
    ax_traj.set_title(f'Trajektoria: {network_name} | Wynik: {final_score:.1f}')
    traj_plot_path = os.path.join(local_storage, f"trajectory_{network_name}.png")
    plt.savefig(traj_plot_path, dpi=150) 
    plt.close(fig_traj)
    
    abs_model_path = "file:///" + os.path.abspath(model_path).replace('\\', '/')
    abs_traj_path = "file:///" + os.path.abspath(traj_plot_path).replace('\\', '/')

    cursor.execute("""INSERT INTO evaluation_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                (network_name, mean_compliant_v, mean_yaw_err, max_yaw, mean_drift, drift_integral, final_score, abs_model_path, "Brak (Optymalizacja Czasowa)", abs_traj_path))
    conn.commit()
    conn.close()
    print(f"[EVALUATION] Ukończono! Wynik: {final_score:.1f} | Średnia V: {mean_compliant_v:.3f} m/s")

class ModelCheckpointCallback(BaseCallback):
    def __init__(self, save_frequency: int, local_path: str, log_dir: str, target_storage: str, timestamp_str: str, verbose=1):
        super().__init__(verbose)
        self.save_frequency = save_frequency
        self.local_path = local_path
        self.log_dir = log_dir
        self.target_storage = target_storage
        self.timestamp_str = timestamp_str
        os.makedirs(self.local_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_frequency == 0:
            base_name = f"{self.timestamp_str}_{self.n_calls}_steps"
            local_model_path = os.path.join(self.local_path, f"{base_name}")
            
            self.model.save(local_model_path)
            
            if self.verbose > 0:
                print(f"\n[CHECKPOINT] Osiągnięto {self.n_calls} kroków. Zapisano model. Uruchamianie ewaluacji.")
            
            local_zip_path = f"{local_model_path}.zip"
            target_file = os.path.join(self.target_storage, f"{base_name}.zip")
            
            try:
                shutil.copy2(local_zip_path, target_file)
                evaluate_model_performance(target_file, self.target_storage)
            except Exception as e:
                print(f"[CHECKPOINT] Błąd zapisu lub ewaluacji: {e}")
                
        return True

class BestModelEvalCallback(EvalCallback):
    def __init__(self, *args, target_storage, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_storage = target_storage
        self.last_best_score = -float('inf')

    def _on_step(self) -> bool:
        parent_result = super()._on_step()
        if self.best_mean_reward > self.last_best_score:
            self.last_best_score = self.best_mean_reward
            local_best_path = os.path.join(self.best_model_save_path, "best_model.zip")
            if os.path.exists(local_best_path):
                target_name = os.path.join(self.target_storage, "BEST_OVERALL_MODEL.zip")
                shutil.copy2(local_best_path, target_name)
                if self.verbose > 0:
                    print(f"\n[EVALUATION] NOWY REKORD: {self.best_mean_reward:.2f}! Zapisano optymalny model.")
        return parent_result

class CurriculumLearningCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_level = -1 

    def _on_step(self) -> bool:
        level = self.num_timesteps // 1_000_000
        multiplier = min(1.0, 0.1 + level * 0.1)

        self.training_env.env_method("set_rotation_drift_multiplier", multiplier)
        
        if level != self.last_level and self.verbose > 0:
            print(f"[CURRICULUM] Awans na poziom {level} w kroku: {self.num_timesteps} | Mnożnik kar zaktualizowany na: {multiplier * 100:.0f}%")
            self.last_level = level
            
        return True
    
class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_buffer_size, verbose=0):
        super().__init__(verbose)
        self.total_buffer_size = total_buffer_size

    def _on_step(self) -> bool:
        if self.num_timesteps % 10_000 == 0:
            print(f"[PROGRESS] Zebrano {self.num_timesteps % self.total_buffer_size} / {self.total_buffer_size} próbek MDP.")
        return True

if __name__ == '__main__':
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    velocities = np.linspace(0.0, 0.3, 30)
    TARGET_GRID = list(itertools.product(angles, velocities))
    
    NUM_ENVS = 8 
    LOCAL_STORAGE_DIR = "./Hexapod_Models_V2/"
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
    print(f"--- ZAINICJOWANO LOKALNĄ PRZESTRZEŃ ROBOCZĄ W: {LOCAL_STORAGE_DIR} ---")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"hexapod_CPG_MATRIX_{timestamp}.zip"
    local_log_folder = f"./hexapod_cpg_logs/PPO_{timestamp}/"
    local_model_folder = f"./intermediate_models_cpg_{timestamp}/"

    print(f"--- INICJALIZACJA {NUM_ENVS} RÓWNOLEGŁYCH ŚRODOWISK ({timestamp}) ---")

    def make_env(rank):
        def _init():
            targets_per_env = len(TARGET_GRID) // NUM_ENVS
            start_idx = rank * targets_per_env
            end_idx = start_idx + targets_per_env
            sub_grid = TARGET_GRID[start_idx:end_idx]
            raw_env = HexapodRLTrainingEnv(render_mode=None, target_grid=sub_grid, enable_noise=True)
            return Monitor(raw_env, os.path.join(local_log_folder, f"env_{rank}"))
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    raw_test_env = HexapodRLTrainingEnv(render_mode=None, target_grid=TARGET_GRID)
    test_env = Monitor(raw_test_env)

    TOTAL_BUFFER_SIZE = len(TARGET_GRID) * MAX_EPISODE_STEPS
    N_STEPS_PPO = TOTAL_BUFFER_SIZE // NUM_ENVS 

    progress_callback = TrainingProgressCallback(total_buffer_size=TOTAL_BUFFER_SIZE)

    PRETRAINED_MODEL_PATH = "" 

    if PRETRAINED_MODEL_PATH != "":
        print(f"--- TRANSFER LEARNING: Wczytywanie modelu z {PRETRAINED_MODEL_PATH} ---")
        training_hyperparams = {
            "n_steps": N_STEPS_PPO,      
            "batch_size": 16000,         
            "n_epochs": 5,             
            "learning_rate": 0.0000001,  
            "clip_range": 0.02,         
            "ent_coef": 0.0,            
            "target_kl": 0.01           
        }
        model = PPO.load(
            PRETRAINED_MODEL_PATH,
            env=env,
            custom_objects=training_hyperparams,
            tensorboard_log="./hexapod_cpg_logs/",
            device="cpu"
        )
    else:
        print("--- FAZA 2: INICJALIZACJA NOWEJ ARCHITEKTURY SIECI NEURONOWEJ ---")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_architecture,
            learning_rate=0.0001,
            n_steps=N_STEPS_PPO,         
            batch_size=1000,             
            n_epochs=10, 
            clip_range=0.15,
            target_kl=0.07,
            ent_coef=0.01,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./hexapod_cpg_logs/",
            device="cpu" 
        )
        
    checkpoint_cb = ModelCheckpointCallback(120001, local_model_folder, local_log_folder, LOCAL_STORAGE_DIR, timestamp)
    eval_cb = BestModelEvalCallback(test_env, best_model_save_path=local_model_folder, log_path=local_log_folder, eval_freq=120001, n_eval_episodes=4, deterministic=True, render=False, target_storage=LOCAL_STORAGE_DIR)
    curriculum_cb = CurriculumLearningCallback(verbose=1)

    print("--- ROZPOCZĘCIE TRENINGU WIELOWĄTKOWEGO ---")
    
    try:
        model.learn(
            total_timesteps=30000000,
            callback=[checkpoint_cb, eval_cb, progress_callback],
            progress_bar=True,
            reset_num_timesteps=False
        )   
    except KeyboardInterrupt:
        print("\n--- TRENING PRZERWANY (CTRL+C). ZAPISYWANIE STANU! ---")
    except Exception as e:
        print(f"\n--- WYSTĄPIŁ BŁĄD KRYTYCZNY: {e}. ZAPISYWANIE STANU! ---")

    print("--- ZAPISYWANIE FINALNEGO MODELU ---")
    model.env = None
    
    try:
        final_model_path = os.path.join(LOCAL_STORAGE_DIR, model_name)
        model.save(final_model_path)
        print(f">>> FINALNY MODEL ZAPISANY POMYŚLNIE: {final_model_path} <<<")
    except Exception as e:
        print(f"Błąd przy zapisie ostatecznego modelu: {e}")

    try:
        env.close()
        test_env.close()
    except Exception:
        pass

    print("PROCES TRENINGOWY ZAKOŃCZONY!")
