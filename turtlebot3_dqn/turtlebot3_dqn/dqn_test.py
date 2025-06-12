#!/usr/bin/env python3
#################################################################################
# DQN Test for TurtleBot3 - 모델 경로 문제 해결
#################################################################################

import collections
import os
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from turtlebot3_msgs.srv import Dqn


class DQNTest(Node):

    def __init__(self, stage, load_episode):
        super().__init__('dqn_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)

        self.state_size = 14
        self.action_size = 5

        self.memory = collections.deque(maxlen=1000000)

        self.model = self.build_model()
        
        # 모델 경로 수정 - 여러 가능한 경로 시도
        model_path = self.find_model_path()
        
        if model_path and os.path.exists(model_path):
            try:
                loaded_model = load_model(
                    model_path, compile=False, custom_objects={'mse': MeanSquaredError()}
                )
                self.model.set_weights(loaded_model.get_weights())
                self.get_logger().info(f'Successfully loaded model from: {model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model from {model_path}: {e}')
                self.get_logger().info('Using randomly initialized model instead')
        else:
            self.get_logger().error(f'Model file not found: stage{self.stage}_episode{self.load_episode}.h5')
            self.get_logger().info('Available model files:')
            self.list_available_models()
            self.get_logger().info('Using randomly initialized model instead')

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.run_test()

    def find_model_path(self):
        """Find model file in multiple possible locations"""
        model_filename = f'stage{self.stage}_episode{self.load_episode}.h5'
        
        # 가능한 경로들 (실제 경로 추가)
        possible_paths = [
            # 1. 실제 모델 파일이 있는 경로
            os.path.join(
                os.path.expanduser('~'), 'turtlebot3_ws', 'src', 
                'turtlebot3_machine_learning', 'turtlebot3_dqn', 'saved_model', 
                model_filename
            ),
            # 2. turtlebot3_drl 경로 (다른 구조인 경우)
            os.path.join(
                os.path.expanduser('~'), 'turtlebot3_ws', 'src', 'turtlebot3_drl', 
                'saved_model', model_filename
            ),
            # 3. 원래 코드의 경로 (install 기준)
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                'saved_model',
                model_filename
            ),
            # 4. 현재 디렉토리 기준
            os.path.join(os.getcwd(), 'saved_model', model_filename),
            # 5. 홈 디렉토리 기준
            os.path.join(os.path.expanduser('~'), 'saved_model', model_filename),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            self.get_logger().info(f'Checking path: {abs_path}')
            if os.path.exists(abs_path):
                return abs_path
        
        return None

    def list_available_models(self):
        """List all available model files"""
        search_dirs = [
            # 실제 모델이 있는 디렉토리
            os.path.join(os.path.expanduser('~'), 'turtlebot3_ws', 'src', 
                        'turtlebot3_machine_learning', 'turtlebot3_dqn', 'saved_model'),
            os.path.join(os.path.expanduser('~'), 'turtlebot3_ws', 'src', 'turtlebot3_drl', 'saved_model'),
            os.path.join(os.getcwd(), 'saved_model'),
            os.path.join(os.path.expanduser('~'), 'saved_model'),
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                self.get_logger().info(f'Directory: {search_dir}')
                try:
                    files = [f for f in os.listdir(search_dir) if f.endswith('.h5')]
                    if files:
                        for file in sorted(files):
                            self.get_logger().info(f'  - {file}')
                    else:
                        self.get_logger().info('  (no .h5 files found)')
                except Exception as e:
                    self.get_logger().info(f'  (error reading directory: {e})')
            else:
                self.get_logger().info(f'Directory not found: {search_dir}')
    def build_model(self):
        model = Sequential()
        model.add(Dense(
            512, input_shape=(self.state_size,),
            activation='relu',
            kernel_initializer='lecun_uniform'
        ))
        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.00025))
        return model

    def get_action(self, state):
        state = numpy.asarray(state)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        action = int(numpy.argmax(q_values[0]))
        
        # 액션 로깅 (가끔씩만)
        if hasattr(self, '_action_count'):
            self._action_count += 1
        else:
            self._action_count = 1
            
        if self._action_count % 50 == 0:
            self.get_logger().info(f'Q-values: {q_values[0]}, Selected action: {action}')
        
        return action

    def run_test(self):
        """Run continuous testing like DDPG"""
        total_goals = 0
        successful_goals = 0
        collision_count = 0
        timeout_count = 0
        
        self.get_logger().info('Starting continuous DQN test')

        while True:
            done = False
            init = True
            score = 0
            local_step = 0
            next_state = []

            time.sleep(1.0)

            while not done:
                local_step += 1
                
                # 첫 번째 액션은 전진, 나머지는 모델 예측
                if local_step == 1:
                    action = 2  # 전진
                else:
                    action = self.get_action(next_state)

                req = Dqn.Request()
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn(
                        'rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    next_state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    score += reward
                    init = False
                    
                    # 주기적 로깅
                    if local_step % 50 == 0:
                        self.get_logger().info(
                            f'Step {local_step}: Action={action}, Reward={reward:.2f}, Score={score:.2f}'
                        )
                        
                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')
                    break

                time.sleep(0.01)
            
            # 에피소드 완료 처리
            if done:
                total_goals += 1
                
                # 결과 분석 (보상 기반 추정)
                if score > 100:  # 성공 추정
                    successful_goals += 1
                    self.get_logger().info(
                        f'🎯 Goal {total_goals} REACHED! Steps: {local_step}, Score: {score:.1f}'
                    )
                elif score < -50:  # 충돌 추정
                    collision_count += 1
                    self.get_logger().info(
                        f'💥 Collision at Goal {total_goals}! Steps: {local_step}, Score: {score:.1f}'
                    )
                else:  # 타임아웃 추정
                    timeout_count += 1
                    self.get_logger().info(
                        f'⏰ Timeout at Goal {total_goals}! Steps: {local_step}, Score: {score:.1f}'
                    )
                
                # 통계 출력
                if total_goals > 0:
                    success_rate = (successful_goals / total_goals) * 100
                    self.get_logger().info(
                        f'📊 Statistics - Total: {total_goals}, '
                        f'Success: {successful_goals} ({success_rate:.1f}%), '
                        f'Collisions: {collision_count}, Timeouts: {timeout_count}'
                    )
                
                # 10개 goal마다 요약
                if total_goals % 10 == 0:
                    self.get_logger().info('=' * 60)
                    self.get_logger().info(f'🏆 {total_goals} GOALS COMPLETED!')
                    self.get_logger().info(f'Success Rate: {success_rate:.1f}%')
                    self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args if args else sys.argv)
    stage = sys.argv[1] if len(sys.argv) > 1 else '1'
    load_episode = sys.argv[2] if len(sys.argv) > 2 else '600'
    
    print(f"Starting DQN test for stage {stage}, episode {load_episode}")
    
    # DQN 테스트 (DDPG가 아님!)
    node = DQNTest(stage, load_episode)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Test interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()