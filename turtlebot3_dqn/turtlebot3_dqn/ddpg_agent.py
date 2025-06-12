#!/usr/bin/env python3
#################################################################################
# DDPG Agent for TurtleBot3 - 핵심 오류만 수정
# 1. Action Scaling 오류 수정
# 2. 하이퍼파라미터 개선
#################################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import collections
import datetime
import json
import math
import random
import sys
import time
import copy

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.optimizers import Adam

from turtlebot3_msgs.srv import Dqn

tf.config.set_visible_devices([], 'GPU')

LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')

# Action indices
LINEAR = 0
ANGULAR = 1

# Environment outcome constants (Environment와 동일)
SUCCESS = 1
COLLISION_WALL = 2
COLLISION_OBSTACLE = 3
TIMEOUT = 4
TUMBLE = 5


class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration - 개선된 파라미터"""
    def __init__(self, action_size, max_sigma=0.3, min_sigma=0.1, decay_period=1000000):  # 개선된 파라미터
        self.action_size = action_size
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_size)

    def get_noise(self, step):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, step / self.decay_period)
        dx = 0.15 * (0.0 - self.state) + sigma * np.random.randn(self.action_size)
        self.state = self.state + dx
        return self.state


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def add_sample(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        actions = np.array(actions).reshape(batch_size, -1)
        rewards = np.array(rewards).reshape(batch_size, -1)
        done = np.array(done).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def get_length(self):
        return len(self.buffer)


class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Network layers with better initialization
        self.fa1 = Dense(hidden_size, activation='relu', 
                        kernel_initializer='he_normal',
                        bias_initializer='zeros')
        self.fa2 = Dense(hidden_size, activation='relu', 
                        kernel_initializer='he_normal',
                        bias_initializer='zeros')
        self.fa3 = Dense(action_size, activation='tanh', 
                        kernel_initializer=tf.keras.initializers.RandomUniform(-0.003, 0.003),
                        bias_initializer='zeros')
        
        # Build the model
        self.build((None, state_size))

    def call(self, states, training=False):
        x1 = self.fa1(states)
        x2 = self.fa2(x1)
        action = self.fa3(x2)
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Network layers
        self.l1 = Dense(hidden_size // 2, activation='relu', kernel_initializer='he_normal')
        self.l2 = Dense(hidden_size // 2, activation='relu', kernel_initializer='he_normal')
        self.l3 = Dense(hidden_size, activation='relu', kernel_initializer='he_normal')
        self.l4 = Dense(1, activation=None, kernel_initializer='glorot_normal')
        
        # Build the model
        self.build([(None, state_size), (None, action_size)])

    def call(self, inputs, training=False):
        states, actions = inputs
        xs = self.l1(states)
        xa = self.l2(actions)
        x = concatenate([xs, xa])
        x = self.l3(x)
        x = self.l4(x)
        return x


class DDPGAgent(Node):
    def __init__(self, stage_num, max_training_episodes):
        super().__init__('ddpg_agent')

        self.stage = int(stage_num)
        self.training = True
        self.state_size = 14
        self.action_size = 2
        self.max_training_episodes = int(max_training_episodes)
        self.hidden_size = 256
        
        # Environment bounds
        self.max_linear_vel = 0.22  
        self.max_angular_vel = 2.84
        
        # 개선된 DDPG hyperparameters
        self.gamma = 0.99
        self.tau = 0.001  # 수정: 더 느린 타겟 업데이트
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.batch_size = 64
        self.buffer_size = 1000000
        self.observe_steps = 10000  # 수정: 더 긴 observation
        self.step_time = 0.01

        # Initialize networks
        self.actor = Actor(self.state_size, self.action_size, self.hidden_size)
        self.actor_target = Actor(self.state_size, self.action_size, self.hidden_size)
        self.critic = Critic(self.state_size, self.action_size, self.hidden_size)
        self.critic_target = Critic(self.state_size, self.action_size, self.hidden_size)
        
        # Optimizers
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
        
        # Initialize target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # 개선된 noise 파라미터
        self.noise = OUNoise(self.action_size, max_sigma=0.3, min_sigma=0.1, decay_period=1000000)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Training state
        self.total_steps = 0
        self.episode = 0

        # Outcome 정보 수신
        self.last_outcome = 0
        self.outcome_sub = self.create_subscription(
            Float32MultiArray, '/episode_outcome', self.outcome_callback, 10)

        # Model saving
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )

        # Logging
        if LOGGING:
            tensorboard_file_name = current_time + ' ddpg_stage' + str(self.stage) + '_reward'
            home_dir = os.path.expanduser('~')
            ddpg_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_ddpg_logs', 'gradient_tape', tensorboard_file_name
            )
            self.ddpg_reward_writer = tf.summary.create_file_writer(ddpg_reward_log_dir)

        # ROS2 interfaces
        self.step_comm_client = self.create_client(Dqn, 'step_comm')
        self.goal_comm_client = self.create_client(Empty, 'goal_comm')
        self.pause_physics_client = self.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = self.create_client(Empty, '/unpause_physics')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.get_logger().info('DDPG Agent initialized successfully')
        self.process()

    def outcome_callback(self, msg):
        """Receive outcome information from environment"""
        if len(msg.data) > 0:
            self.last_outcome = int(msg.data[0])

    def hard_update(self, target, source):
        """Hard update: copy weights from source to target"""
        target.set_weights(source.get_weights())

    def soft_update(self, target, source, tau):
        """Soft update: target = tau * source + (1 - tau) * target"""
        target_weights = target.get_weights()
        source_weights = source.get_weights()
        
        for i in range(len(target_weights)):
            target_weights[i] = tau * source_weights[i] + (1 - tau) * target_weights[i]
        
        target.set_weights(target_weights)

    def get_action(self, state, is_training, step):
        """Get action from actor network"""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state, training=False)[0].numpy()
        
        if step % 50 == 0:
            self.get_logger().info(f'Raw actor output: [{action[0]:.3f}, {action[1]:.3f}]')
        
        if is_training:
            noise = self.noise.get_noise(step)
            action = np.clip(action + noise, -1.0, 1.0)
            
            if step % 50 == 0:
                self.get_logger().info(f'After noise: [{action[0]:.3f}, {action[1]:.3f}]')
        
        return action

    def get_action_random(self):
        """Get random action for exploration"""
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0) for _ in range(self.action_size)]

    def scale_action(self, action):
        """Scale action from [-1, 1] to actual velocity ranges - 수정된 버전"""
        # 수정: 선속도를 [0, max_linear_vel] 범위로 매핑 (후진 방지)
        linear_vel = (action[LINEAR] + 1.0) / 2.0 * self.max_linear_vel
        angular_vel = action[ANGULAR] * self.max_angular_vel
        
        # 안전을 위해 클리핑
        linear_vel = np.clip(linear_vel, 0.0, self.max_linear_vel)
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        return [linear_vel, angular_vel]

    def train_step(self):
        """Single training step"""
        if self.replay_buffer.get_length() < self.batch_size:
            return [0.0, 0.0]
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Train Critic
        with tf.GradientTape() as tape:
            # Target Q-values
            action_next = self.actor_target(next_states, training=False)
            Q_next = self.critic_target([next_states, action_next], training=False)
            Q_target = rewards + (1 - dones) * self.gamma * Q_next
            
            # Current Q-values
            Q = self.critic([states, actions], training=True)
            
            # Critic loss
            loss_critic = tf.reduce_mean(tf.square(Q - tf.stop_gradient(Q_target)))
        
        # Update critic
        critic_grads = tape.gradient(loss_critic, self.critic.trainable_variables)
        critic_grads = [tf.clip_by_norm(grad, 2.0) for grad in critic_grads]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Train Actor
        with tf.GradientTape() as tape:
            pred_actions = self.actor(states, training=True)
            loss_actor = -tf.reduce_mean(self.critic([states, pred_actions], training=False))
        
        # Update actor
        actor_grads = tape.gradient(loss_actor, self.actor.trainable_variables)
        actor_grads = [tf.clip_by_norm(grad, 2.0) for grad in actor_grads]
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        return [loss_critic.numpy(), loss_actor.numpy()]

    def init_episode(self, force_reset=False):
        """Initialize episode - 이중 리셋 방지"""
        if force_reset:
            # 강제 리셋 (실패 시 또는 첫 에피소드만)
            while not self.goal_comm_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('goal_comm service not available, waiting...')
            
            future = self.goal_comm_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().info('Environment forcefully reset')
            
            # 강제 리셋 후 상태 초기화
            req = Dqn.Request()
            req.action = 254  # Reset signal
            
            future = self.step_comm_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                return list(future.result().state)
            return [0.0] * self.state_size
        
        else:
            # 성공 시: 리셋하지 않고 현재 상태만 가져오기
            # Environment에서 이미 새 Goal을 설정했으므로 추가 리셋 불필요
            self.get_logger().info('Continuing without environment reset')
            
            # 현재 상태만 요청 (리셋 신호 없이)
            req = Dqn.Request()
            req.action = 253  # 새로운 신호: 상태만 요청
            
            future = self.step_comm_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                return list(future.result().state)
            return [0.0] * self.state_size

    def step(self, action_scaled):
        """Take a step in the environment"""
        req = Dqn.Request()
        req.action = 255  # Special flag for continuous action
        
        # Publish velocities via topic
        msg = Float32MultiArray()
        msg.data = [float(action_scaled[0]), float(action_scaled[1])]
        self.action_pub.publish(msg)

        while not self.step_comm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('step_comm service not available, waiting...')

        future = self.step_comm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            result = future.result()
            next_state = list(result.state)
            reward = result.reward
            done = result.done
            
            # outcome은 에피소드 완료 시에만 유효
            outcome = self.last_outcome if done else 0
            distance_traveled = 0.0
            
            return next_state, reward, done, outcome, distance_traveled
        else:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return [], 0.0, True, COLLISION_OBSTACLE, 0.0

    def pause_simulation(self):
        """Pause Gazebo simulation"""
        if not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            return
        future = self.pause_physics_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

    def unpause_simulation(self):
        """Unpause Gazebo simulation"""
        if not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            return
        future = self.unpause_physics_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

    def process(self):
        """Main training loop - 이중 리셋 방지"""
        self.pause_simulation()
        
        # 첫 에피소드는 강제 리셋
        force_reset = True
        
        while self.episode < self.max_training_episodes:
            # 에피소드 초기화
            state = self.init_episode(force_reset)
            episode_done = False
            step = 0
            reward_sum = 0
            loss_critic_sum = 0
            loss_actor_sum = 0
            action_past = [0.0, 0.0]
            outcome = 0
            distance_traveled = 0.0
            
            self.unpause_simulation()
            time.sleep(0.5)
            episode_start = time.perf_counter()
            
            while not episode_done:
                # Get action
                if self.training and self.total_steps < self.observe_steps:
                    action = self.get_action_random()
                else:
                    action = self.get_action(state, self.training, step)
                
                # Scale action to actual velocities
                action_scaled = self.scale_action(action)
                
                # Take step
                next_state, reward, episode_done, step_outcome, step_distance = self.step(action_scaled)
                action_past = copy.deepcopy(action_scaled)
                reward_sum += reward
                
                # outcome 업데이트
                if episode_done:
                    outcome = step_outcome
                distance_traveled += step_distance
                
                # Store experience
                if self.training:
                    self.replay_buffer.add_sample(state, action, reward, next_state, episode_done)
                    
                    # Train
                    if self.replay_buffer.get_length() >= self.batch_size:
                        loss_c, loss_a = self.train_step()
                        loss_critic_sum += loss_c
                        loss_actor_sum += loss_a
                
                # Publish action info
                msg = Float32MultiArray()
                msg.data = [action_scaled[0], action_scaled[1], float(reward_sum), float(reward)]
                self.action_pub.publish(msg)
                
                state = copy.deepcopy(next_state)
                step += 1
                self.total_steps += 1
                time.sleep(self.step_time)
            
            # Episode finished
            self.pause_simulation()
            duration = time.perf_counter() - episode_start
            
            # 에피소드 완료 후 리셋 여부 결정
            if outcome in [COLLISION_OBSTACLE, COLLISION_WALL, TIMEOUT, TUMBLE]:
                force_reset = True
                self.get_logger().info(f'Episode failed (outcome: {outcome}), will reset environment next episode')
            elif outcome == SUCCESS:
                force_reset = False
                self.get_logger().info('Episode succeeded, will continue without reset next episode')
                # 성공 시 약간의 대기 시간으로 Environment의 Goal 업데이트 완료 대기
                time.sleep(0.1)
            else:
                force_reset = True  # 알 수 없는 상태는 안전하게 리셋
                self.get_logger().info('Unknown outcome, will reset environment next episode')
            
            self.finish_episode(step, duration, outcome, distance_traveled, 
                              reward_sum, loss_critic_sum, loss_actor_sum)

    def finish_episode(self, step, duration, outcome, distance_traveled, reward_sum, 
                      loss_critic, loss_actor):
        """Finish episode and log results"""
        if self.total_steps < self.observe_steps:
            print(f"Observe phase: {self.total_steps}/{self.observe_steps} steps")
            return
        
        self.episode += 1
        
        # Log episode results
        print(f"Episode: {self.episode:<5} Reward: {reward_sum:<8.1f} Steps: {step:<6} "
              f"Total Steps: {self.total_steps:<7} Time: {duration:<6.2f}s Outcome: {outcome}")
        
        if self.training:
            # Publish results
            msg = Float32MultiArray()
            msg.data = [float(reward_sum), 0.0]
            self.result_pub.publish(msg)
            
            # TensorBoard logging
            if LOGGING:
                with self.ddpg_reward_writer.as_default():
                    tf.summary.scalar('episode_reward', reward_sum, step=self.episode)
                    tf.summary.scalar('episode_steps', step, step=self.episode)
                    tf.summary.scalar('outcome', outcome, step=self.episode)
                    if step > 0:
                        tf.summary.scalar('critic_loss', loss_critic / step, step=self.episode)
                        tf.summary.scalar('actor_loss', loss_actor / step, step=self.episode)
            
            # Save model periodically
            if self.episode % 100 == 0:
                self.save_models()

    def save_models(self):
        """Save models - 새 TensorFlow 버전에 맞게 .weights.h5 사용"""
        if not os.path.exists(self.model_dir_path):
            os.makedirs(self.model_dir_path)
            
        # 새 버전 요구사항: .weights.h5 확장자 사용
        actor_path = os.path.join(self.model_dir_path, f'ddpg_actor_stage{self.stage}_ep{self.episode}.weights.h5')
        critic_path = os.path.join(self.model_dir_path, f'ddpg_critic_stage{self.stage}_ep{self.episode}.weights.h5')
        
        try:
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            self.get_logger().info(f'Models saved at episode {self.episode}')
        except Exception as e:
            self.get_logger().error(f'Failed to save models: {e}')
    

def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    
    rclpy.init(args=args)
    ddpg_agent = DDPGAgent(stage_num, max_training_episodes)
    
    try:
        rclpy.spin(ddpg_agent)
    except KeyboardInterrupt:
        ddpg_agent.get_logger().info('Training interrupted by user')
    finally:
        ddpg_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()