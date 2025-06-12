#!/usr/bin/env python3
#################################################################################
# DDPG Environment for TurtleBot3 - Reward Function 파라미터 불일치 수정
# 2. Reward Function 파라미터 불일치 수정
#################################################################################

import math
import os

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal

# Reward function constants
REWARD_FUNCTION = "A"
COLLISION_OBSTACLE = 3
COLLISION_WALL = 2
TUMBLE = 5
SUCCESS = 1
TIMEOUT = 4
RESULTS_NUM = 6

ROS_DISTRO = os.environ.get('ROS_DISTRO')

# Reward function globals
goal_dist_initial = 0
reward_function_internal = None

def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance, prev_distance_to_goal=None):
    return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance, prev_distance_to_goal)

def get_reward_A(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist, prev_goal_dist=None):
    """수정된 reward function - 파라미터 일치"""
    # [-3.14, 0]
    r_yaw = -1 * abs(goal_angle)
    # [-4, 0]
    r_vangular = -1 * (action_angular**2)
    # [-1, 1]
    r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1
    # [-20, 0]
    if min_obstacle_dist < 0.22:
        r_obstacle = -20
    else:
        r_obstacle = 0
    # [-2 * (2.2^2), 0]
    r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)
    
    # 추가: Progress reward (이전 거리가 있을 경우)
    r_progress = 0
    if prev_goal_dist is not None:
        progress = prev_goal_dist - goal_dist
        r_progress = progress * 50.0  # progress에 대한 보상
    
    reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular + r_progress - 1
    
    if succeed == SUCCESS:
        reward += 2500
    elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
        reward -= 2000
    
    return float(reward)

def reward_initalize(init_distance_to_goal):
    global goal_dist_initial, reward_function_internal
    goal_dist_initial = init_distance_to_goal
    function_name = "get_reward_" + REWARD_FUNCTION
    reward_function_internal = globals()[function_name]
    if reward_function_internal == None:
        quit(f"Error: reward function {function_name} does not exist")


class DDPGEnvironment(Node):

    def __init__(self):
        super().__init__('ddpg_environment')
        self.goal_pose_x = 0.5
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 2  # continuous action: [linear_vel, angular_vel]
        self.max_step = 1000  # 수정: 더 짧은 에피소드

        self.done = False
        self.fail = False
        self.succeed = False
        self.current_outcome = 0  # 현재 에피소드 결과 추적

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.prev_goal_distance = 1.0  # 수정: 이전 거리 추적 추가
        self.scan_ranges = []
        self.front_ranges = []
        self.min_obstacle_distance = 10.0

        self.local_step = 0
        self.stop_cmd_vel_timer = None
        
        # 연속 행동 저장 변수
        self.last_continuous_action = [0.1, 0.0]  # [linear_vel, angular_vel]
        
        # 연속 행동 공간 설정
        self.max_linear_vel = 0.22   # TurtleBot3 최대 선속도
        self.max_angular_vel = 2.84  # TurtleBot3 최대 각속도

        qos = QoSProfile(depth=10)

        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        # Outcome 정보 전달용 publisher 추가
        self.outcome_pub = self.create_publisher(Float32MultiArray, '/episode_outcome', 10)

        # Subscribe to continuous action topic
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/get_action',
            self.action_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_sub_callback,
            qos_profile_sensor_data
        )

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            'task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            'task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            'initialize_env',
            callback_group=self.clients_callback_group
        )

        # Service interfaces matching the improved agent
        self.step_comm_service = self.create_service(
            Dqn,
            'step_comm',
            self.step_comm_callback
        )
        self.goal_comm_service = self.create_service(
            Empty,
            'goal_comm',
            self.goal_comm_callback
        )

        self.get_logger().info('DDPG Environment initialized')

    def action_callback(self, msg):
        """Receive continuous action from topic"""
        if len(msg.data) >= 2:
            self.last_continuous_action = [msg.data[0], msg.data[1]]

    def goal_comm_callback(self, request, response):
        """Handle goal communication - 환경 초기화"""
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'service for initialize the environment is not available, waiting ...'
            )
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal.success:
            self.get_logger().error('initialize environment request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(
                f'Environment initialized - Goal at [{self.goal_pose_x:.3f}, {self.goal_pose_y:.3f}]'
            )
        return response

    def step_comm_callback(self, request, response):
        """Handle step communication - 이중 리셋 방지"""
        action = request.action
        
        if action == 254:  # Reset/initialization signal (강제 리셋)
            state = self.calculate_state()
            self.init_goal_distance = state[0] if state else 1.0
            self.prev_goal_distance = self.init_goal_distance  # 수정: 이전 거리 초기화
            reward_initalize(self.init_goal_distance)
            self.local_step = 0
            self.current_outcome = 0
            response.state = state if state else [0.0] * 14
            response.reward = 0.0
            response.done = False
            self.get_logger().info('Environment FORCE reset')
            return response
        
        elif action == 253:  # 새로운 신호: 상태만 요청 (리셋 없이)
            state = self.calculate_state_without_reset()
            # 수정: 새 에피소드 시작시 거리 초기화
            if state:
                self.prev_goal_distance = state[0]
                self.init_goal_distance = state[0]
                reward_initalize(self.init_goal_distance)
            response.state = state if state else [0.0] * 14
            response.reward = 0.0
            response.done = False
            self.get_logger().info('State requested without reset')
            return response
        
        elif action == 255:  # Continuous action signal
            linear_vel, angular_vel = self.last_continuous_action
            self.execute_continuous_action(linear_vel, angular_vel)
        
        else:  # Discrete action (backward compatibility)
            self.execute_discrete_action(action)

        # Calculate next state and reward
        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        response.done = self.done

        # Send outcome information when episode is done
        if self.done:
            outcome_msg = Float32MultiArray()
            outcome_msg.data = [float(self.current_outcome)]
            self.outcome_pub.publish(outcome_msg)
            
            self.done = False
            self.succeed = False
            self.fail = False
            self.current_outcome = 0

        return response

    def call_task_succeed(self):
        """성공 시: Goal만 새 위치로 이동"""
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('Task succeeded - Goal moved to new position')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        """실패 시: 로봇과 Goal 모두 리셋"""
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('Task failed - Robot and Goal reset')
        else:
            self.get_logger().error('task failed service call failed')

    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []

        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = scan.ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                self.front_ranges.append(distance)

        self.min_obstacle_distance = min(self.scan_ranges) if self.scan_ranges else 10.0

    def odom_sub_callback(self, msg):
        # 수정: 이전 거리 저장
        prev_distance = self.goal_distance if hasattr(self, 'goal_distance') else 1.0
        
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        # 수정: 이전 거리 업데이트
        self.prev_goal_distance = prev_distance
        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state_without_reset(self):
        """현재 상태만 계산 (터미널 조건 체크 없이)"""
        if not self.front_ranges:
            return [1.0, 0.0] + [3.5] * 12
        
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        
        # Ensure we have exactly 12 front range values
        front_ranges_padded = self.front_ranges[:12]
        while len(front_ranges_padded) < 12:
            front_ranges_padded.append(3.5)
            
        for var in front_ranges_padded:
            state.append(float(var))
        
        # local_step은 증가시키지 않음 (새 에피소드 시작 대기)
        return state
    
    def calculate_state(self):
        """Calculate current state - 기존 로직 유지"""
        if not self.front_ranges:
            return [1.0, 0.0] + [3.5] * 12
        
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        
        front_ranges_padded = self.front_ranges[:12]
        while len(front_ranges_padded) < 12:
            front_ranges_padded.append(3.5)
            
        for var in front_ranges_padded:
            state.append(float(var))
            
        self.local_step += 1

        # Check terminal conditions
        if self.goal_distance < 0.20:
            self.get_logger().info('Goal Reached!')
            self.succeed = SUCCESS
            self.current_outcome = SUCCESS
            self.done = True
            self.stop_robot()
            self.local_step = 0
            self.call_task_succeed()  # Goal만 새 위치로 이동

        if self.min_obstacle_distance < 0.15:
            self.get_logger().info('Collision detected!')
            self.fail = COLLISION_OBSTACLE
            self.current_outcome = COLLISION_OBSTACLE
            self.done = True
            self.stop_robot()
            self.local_step = 0
            self.call_task_failed()  # 로봇 + Goal 모두 리셋

        if self.local_step >= self.max_step:
            self.get_logger().info('Episode timeout!')
            self.fail = TIMEOUT
            self.current_outcome = TIMEOUT
            self.done = True
            self.stop_robot()
            self.local_step = 0
            self.call_task_failed()  # 로봇 + Goal 모두 리셋

        return state

    def calculate_reward(self):
        """Calculate reward using the new reward function - 수정된 파라미터"""
        # Determine succeed status
        succeed_status = 0  # Default: continuing
        if hasattr(self, 'succeed') and self.succeed:
            succeed_status = self.succeed
        elif hasattr(self, 'fail') and self.fail:
            succeed_status = self.fail
        
        # Get current action (last executed action)
        action_linear = self.last_continuous_action[0]
        action_angular = self.last_continuous_action[1]
        
        # 수정: prev_distance_to_goal 파라미터 추가
        reward = get_reward(
            succeed_status,
            action_linear,
            action_angular,
            self.goal_distance,
            self.goal_angle,
            self.min_obstacle_distance,
            self.prev_goal_distance  # 수정: 이전 거리 파라미터 추가
        )

        if self.local_step % 100 == 0:
            self.get_logger().info(
                f'Reward: {reward:.2f}, Goal Dist: {self.goal_distance:.2f}, '
                f'Goal Angle: {self.goal_angle:.2f}, Min Obstacle: {self.min_obstacle_distance:.2f}'
            )

        return reward

    def execute_continuous_action(self, linear_vel, angular_vel):
        """Execute continuous action directly"""
        # Clip velocities to safe ranges
        linear_vel = numpy.clip(linear_vel, 0.0, self.max_linear_vel)
        angular_vel = numpy.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        if self.local_step % 100 == 0:
            self.get_logger().info(
                f'Executing: Linear={linear_vel:.3f}, Angular={angular_vel:.3f}'
            )
        
        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = float(linear_vel)
            msg.angular.z = float(angular_vel)
        else:
            msg = TwistStamped()
            msg.twist.linear.x = float(linear_vel)
            msg.twist.angular.z = float(angular_vel)

        self.cmd_vel_pub.publish(msg)

    def execute_discrete_action(self, action):
        """Execute discrete action (backward compatibility)"""
        angular_velocities = [1.5, 0.75, 0.0, -0.75, -1.5]
        if 0 <= action < len(angular_velocities):
            angular_vel = angular_velocities[action]
            linear_vel = 0.15
        else:
            angular_vel = 0.0
            linear_vel = 0.1
        
        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = linear_vel
            msg.angular.z = angular_vel
        else:
            msg = TwistStamped()
            msg.twist.linear.x = linear_vel
            msg.twist.angular.z = angular_vel

        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop the robot"""
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    ddpg_environment = DDPGEnvironment()
    try:
        rclpy.spin(ddpg_environment)
    except KeyboardInterrupt:
        ddpg_environment.get_logger().info('Environment stopped by user')
    finally:
        ddpg_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()