#!/usr/bin/env python3
#################################################################################
# DDPG Test for TurtleBot3 - 개선된 구조에 맞춤
# 연속 제어 테스트 및 현재 Environment 인터페이스 호환
#################################################################################

import os
import sys
import time
import copy

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from turtlebot3_msgs.srv import Dqn

# Environment outcome constants
SUCCESS = 1
COLLISION_WALL = 2
COLLISION_OBSTACLE = 3
TIMEOUT = 4
TUMBLE = 5


class Actor(tf.keras.Model):
    """Actor network for loading trained model"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Network layers (훈련된 모델과 동일한 구조)
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


class DDPGTest(Node):
    def __init__(self, stage, load_episode):
        super().__init__('ddpg_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)

        self.state_size = 14
        self.action_size = 2  # [linear_velocity, angular_velocity]
        self.hidden_size = 256
        
        # Action bounds for TurtleBot3
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.84

        # Outcome 정보 수신
        self.last_outcome = 0
        self.outcome_sub = self.create_subscription(
            Float32MultiArray, '/episode_outcome', self.outcome_callback, 10)

        # Load trained actor model
        self.actor_model = self.load_actor_model()

        # ROS2 interfaces (개선된 구조에 맞춤)
        self.step_comm_client = self.create_client(Dqn, 'step_comm')
        self.goal_comm_client = self.create_client(Empty, 'goal_comm')
        
        # Action publisher (Environment와 통신용)
        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)

        self.get_logger().info('DDPG Test initialized successfully')
        self.run_test()

    def outcome_callback(self, msg):
        """Receive outcome information from environment"""
        if len(msg.data) > 0:
            self.last_outcome = int(msg.data[0])

    def load_actor_model(self):
        """Load the trained actor model weights - .weights.h5 사용"""
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        
        # .weights.h5 확장자 사용
        model_path = os.path.join(
            model_dir, f'ddpg_actor_stage{self.stage}_ep{self.load_episode}.weights.h5'
        )

        if not os.path.exists(model_path):
            self.get_logger().error(f'Model file not found: {model_path}')
            return None

        try:
            # Create actor model with same architecture
            actor = Actor(self.state_size, self.action_size, self.hidden_size)
            
            # Build the model first (중요!)
            dummy_input = tf.zeros((1, self.state_size))
            _ = actor(dummy_input)
            
            # Load weights
            actor.load_weights(model_path)
            
            self.get_logger().info(f'Successfully loaded actor weights from: {model_path}')
            return actor
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None

    def get_action(self, state):
        """Get continuous action from the trained actor network"""
        if self.actor_model is None:
            self.get_logger().error('Actor model not loaded!')
            return self.get_default_action()

        try:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            action = self.actor_model(state, training=False)[0].numpy()
            
            # Clip actions to [-1, 1] range
            action = np.clip(action, -1.0, 1.0)
            
            return action
        except Exception as e:
            self.get_logger().error(f'Error in get_action: {e}')
            return self.get_default_action()

    def get_default_action(self):
        """Return default action when model fails"""
        return np.array([0.5, 0.0])  # 중간 속도로 직진

    def scale_action(self, action):
        """Scale action from [-1, 1] to actual velocity ranges"""
        linear_vel = action[0] * self.max_linear_vel
        angular_vel = action[1] * self.max_angular_vel
        
        # 안전을 위해 선속도는 최소 0으로 제한
        linear_vel = np.clip(linear_vel, 0.0, self.max_linear_vel)
        
        return [linear_vel, angular_vel]

    def init_episode(self):
        """Initialize new episode"""
        # Wait for new goal
        while not self.goal_comm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('goal_comm service not available, waiting...')
        
        future = self.goal_comm_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)
        
        # Get initial state
        req = Dqn.Request()
        req.action = 254  # Reset signal
        
        future = self.step_comm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            return list(future.result().state)
        return [0.0] * self.state_size

    def step(self, action_scaled):
        """Take a step in the environment"""
        req = Dqn.Request()
        req.action = 255  # Continuous action signal
        
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
            
            return next_state, reward, done, outcome
        else:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return [], 0.0, True, COLLISION_OBSTACLE

    def run_test(self):
        """Run continuous testing - goal 도달 시 새 goal로 이어짐"""
        total_goals = 0
        successful_goals = 0
        collision_count = 0
        timeout_count = 0
        
        # 첫 번째 goal 설정
        state = self.init_episode()
        current_step = 0
        current_score = 0
        
        self.get_logger().info('Starting continuous DDPG test')
        time.sleep(1.0)

        while True:  # 무한 테스트 루프
            current_step += 1
            
            # Get action from trained model
            continuous_action = self.get_action(state)
            action_scaled = self.scale_action(continuous_action)
            
            # Log action periodically
            if current_step % 50 == 0:
                self.get_logger().info(
                    f'Step {current_step}: Action -> '
                    f'Raw: [{continuous_action[0]:.3f}, {continuous_action[1]:.3f}], '
                    f'Scaled: [{action_scaled[0]:.3f}, {action_scaled[1]:.3f}]'
                )

            # Take step
            next_state, reward, done, outcome = self.step(action_scaled)
            current_score += reward
            
            if current_step % 50 == 0:
                self.get_logger().info(
                    f'Step {current_step}: Reward: {reward:.3f}, Current Score: {current_score:.3f}'
                )
            
            state = copy.deepcopy(next_state)
            
            # Goal 상태 처리
            if done:
                total_goals += 1
                outcome_str = self.get_outcome_string(outcome)
                
                if outcome == SUCCESS:
                    successful_goals += 1
                    self.get_logger().info(
                        f'🎯 Goal {total_goals} REACHED! '
                        f'Steps: {current_step}, Score: {current_score:.1f}'
                    )
                    
                    # 성공 시: 새 goal로 이어서 계속 (리셋 없이)
                    # Environment에서 이미 새 goal을 설정했으므로 계속 진행
                    current_step = 0
                    current_score = 0
                    
                elif outcome in [COLLISION_OBSTACLE, COLLISION_WALL]:
                    collision_count += 1
                    self.get_logger().info(
                        f'💥 Collision at Goal {total_goals}! '
                        f'Steps: {current_step}, Score: {current_score:.1f}'
                    )
                    
                    # 충돌 시: 환경 리셋 후 새 시작
                    state = self.init_episode()
                    current_step = 0
                    current_score = 0
                    
                elif outcome == TIMEOUT:
                    timeout_count += 1
                    self.get_logger().info(
                        f'⏰ Timeout at Goal {total_goals}! '
                        f'Steps: {current_step}, Score: {current_score:.1f}'
                    )
                    
                    # 타임아웃 시: 환경 리셋 후 새 시작
                    state = self.init_episode()
                    current_step = 0
                    current_score = 0
                
                # 통계 출력
                if total_goals > 0:
                    success_rate = (successful_goals / total_goals) * 100
                    self.get_logger().info(
                        f'📊 Statistics - Total Goals: {total_goals}, '
                        f'Success: {successful_goals} ({success_rate:.1f}%), '
                        f'Collisions: {collision_count}, Timeouts: {timeout_count}'
                    )
                
                # 10개 goal마다 요약 출력
                if total_goals % 10 == 0:
                    self.get_logger().info('=' * 60)
                    self.get_logger().info(f'🏆 {total_goals} GOALS COMPLETED!')
                    self.get_logger().info(f'Success Rate: {success_rate:.1f}%')
                    self.get_logger().info(f'Collisions: {collision_count}')
                    self.get_logger().info(f'Timeouts: {timeout_count}')
                    self.get_logger().info('=' * 60)
            
            time.sleep(0.01)

    def get_outcome_string(self, outcome):
        """Convert outcome number to string"""
        outcome_map = {
            0: "CONTINUING",
            SUCCESS: "SUCCESS",
            COLLISION_WALL: "COLLISION_WALL",
            COLLISION_OBSTACLE: "COLLISION_OBSTACLE", 
            TIMEOUT: "TIMEOUT",
            TUMBLE: "TUMBLE"
        }
        return outcome_map.get(outcome, f"UNKNOWN({outcome})")


def main(args=None):
    rclpy.init(args=args if args else sys.argv)
    
    stage = sys.argv[1] if len(sys.argv) > 1 else '1'
    load_episode = sys.argv[2] if len(sys.argv) > 2 else '600'
    
    node = DDPGTest(stage, load_episode)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Test interrupted by user')
    except Exception as e:
        node.get_logger().error(f'Test failed with exception: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()