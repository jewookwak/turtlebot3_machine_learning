#!/usr/bin/env python
#################################################################################
# DDPG Result Graph for TurtleBot3 - Continuous Control Results Visualization
# Based on DQN result_graph with DDPG-specific adaptations
#################################################################################

import signal
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QWidget
import pyqtgraph
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class GraphSubscriber(Node):

    def __init__(self, window):
        super().__init__('ddpg_graph')

        self.window = window

        # DDPG result subscription
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/result',
            self.data_callback,
            10
        )
        
        # Action monitoring subscription (for additional insights)
        self.action_subscription = self.create_subscription(
            Float32MultiArray,
            '/get_action',
            self.action_callback,
            10
        )

    def data_callback(self, msg):
        self.window.receive_result_data(msg)
    
    def action_callback(self, msg):
        self.window.receive_action_data(msg)


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle('DDPG Results - Continuous Control')
        self.setGeometry(50, 50, 800, 900)

        # Episode tracking
        self.ep = []
        self.count = 1

        # DDPG-specific data tracking
        self.rewards = []
        
        # Action tracking for analysis (수정됨)
        self.recent_linear_vels = []
        self.recent_angular_vels = []
        self.episode_avg_linear = []  # 에피소드별 평균 선속도
        self.episode_avg_angular = []  # 에피소드별 평균 각속도
        self.avg_linear_vel = 0.0
        self.avg_angular_vel = 0.0

        # Statistics
        self.max_reward = float('-inf')
        self.min_reward = float('inf')
        self.total_episodes = 0
        self.success_episodes = 0

        self.setup_ui()

        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Statistics panel
        stats_layout = QHBoxLayout()
        
        self.stats_label = QLabel("DDPG Training Statistics")
        self.stats_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.episode_label = QLabel("Episodes: 0")
        self.reward_label = QLabel("Avg Reward: 0.0")
        self.success_label = QLabel("Success Rate: 0%")
        self.action_label = QLabel("Avg Action: [0.0, 0.0]")
        
        stats_layout.addWidget(self.episode_label)
        stats_layout.addWidget(self.reward_label)
        stats_layout.addWidget(self.success_label)
        stats_layout.addWidget(self.action_label)
        
        layout.addWidget(self.stats_label)
        layout.addLayout(stats_layout)

        # Plots
        self.plot_rewards()
        self.plot_actions()
        
        # Add plots to layout
        layout.addWidget(self.rewardsPlt)
        layout.addWidget(self.actionsPlt)

        self.show()

    def plot_rewards(self):
        """Setup reward plot"""
        self.rewardsPlt = pyqtgraph.PlotWidget(title='Episode Rewards (DDPG)')
        self.rewardsPlt.setLabel('left', 'Reward')
        self.rewardsPlt.setLabel('bottom', 'Episode')
        self.rewardsPlt.setMinimumHeight(300)

    def plot_actions(self):
        """Setup action analysis plot"""
        self.actionsPlt = pyqtgraph.PlotWidget(title='Average Actions per Episode')
        self.actionsPlt.setLabel('left', 'Velocity')
        self.actionsPlt.setLabel('bottom', 'Episode')
        self.actionsPlt.setMinimumHeight(300)
        
        # Add legend
        self.actionsPlt.addLegend()

    def receive_result_data(self, msg):
        """Process result data from DDPG agent"""
        if len(msg.data) >= 1:
            episode_reward = msg.data[0]
            
            self.rewards.append(episode_reward)
            self.ep.append(self.count)
            self.count += 1
            self.total_episodes += 1
            
            # Update statistics
            self.max_reward = max(self.max_reward, episode_reward)
            self.min_reward = min(self.min_reward, episode_reward)
            
            # Estimate success (high reward threshold)
            if episode_reward > 200:
                self.success_episodes += 1
            
            # 에피소드별 평균 액션 계산 (수정됨)
            if len(self.recent_linear_vels) > 0:
                avg_linear = sum(self.recent_linear_vels) / len(self.recent_linear_vels)
                avg_angular = sum(self.recent_angular_vels) / len(self.recent_angular_vels)
                
                # 에피소드별 평균 저장
                self.episode_avg_linear.append(avg_linear)
                self.episode_avg_angular.append(avg_angular)
                
                # 전체 평균 업데이트
                self.avg_linear_vel = avg_linear
                self.avg_angular_vel = avg_angular
                
                # 다음 에피소드를 위해 클리어
                self.recent_linear_vels.clear()
                self.recent_angular_vels.clear()
            else:
                # 데이터가 없으면 0으로 설정
                self.episode_avg_linear.append(0.0)
                self.episode_avg_angular.append(0.0)

    def receive_action_data(self, msg):
        """Process action data for analysis"""
        if len(msg.data) >= 2:
            linear_vel = msg.data[0]
            angular_vel = msg.data[1]
            
            # 현재 에피소드의 액션들 누적
            self.recent_linear_vels.append(linear_vel)
            self.recent_angular_vels.append(angular_vel)
            
            # 메모리 관리 (너무 많이 쌓이지 않도록)
            if len(self.recent_linear_vels) > 1000:
                self.recent_linear_vels.pop(0)
                self.recent_angular_vels.pop(0)

    def update_statistics(self):
        """Update statistics labels"""
        if self.total_episodes > 0:
            avg_reward = sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
            success_rate = (self.success_episodes / self.total_episodes) * 100
            
            self.episode_label.setText(f"Episodes: {self.total_episodes}")
            self.reward_label.setText(f"Avg Reward: {avg_reward:.1f}")
            self.success_label.setText(f"Success Rate: {success_rate:.1f}%")
            self.action_label.setText(f"Avg Action: [{self.avg_linear_vel:.3f}, {self.avg_angular_vel:.3f}]")

    def update_plots(self):
        """Update all plots - 실제 데이터 사용"""
        if len(self.ep) > 0 and len(self.rewards) > 0:
            # Update reward plot
            self.rewardsPlt.clear()
            self.rewardsPlt.plot(self.ep, self.rewards, pen='r', name='Episode Reward')
            
            # Add moving average if enough data
            if len(self.rewards) > 10:
                window_size = min(20, len(self.rewards))
                moving_avg = []
                for i in range(len(self.rewards)):
                    start_idx = max(0, i - window_size + 1)
                    avg = sum(self.rewards[start_idx:i+1]) / (i - start_idx + 1)
                    moving_avg.append(avg)
                
                self.rewardsPlt.plot(self.ep, moving_avg, pen='b', name='Moving Average')
            
            # Update action plot - 실제 데이터 사용
            self.actionsPlt.clear()
            if len(self.episode_avg_linear) > 0 and len(self.episode_avg_angular) > 0:
                # 실제 에피소드별 평균 액션 사용
                episodes_with_data = list(range(1, len(self.episode_avg_linear) + 1))
                
                self.actionsPlt.plot(
                    episodes_with_data, 
                    self.episode_avg_linear, 
                    pen='g', 
                    name='Avg Linear Vel'
                )
                self.actionsPlt.plot(
                    episodes_with_data, 
                    self.episode_avg_angular, 
                    pen='m', 
                    name='Avg Angular Vel'
                )
            else:
                # 데이터가 없으면 빈 그래프
                self.actionsPlt.plot([], [], pen='g', name='Avg Linear Vel')
                self.actionsPlt.plot([], [], pen='m', name='Avg Angular Vel')

    def update(self):
        """Main update function called by timer"""
        self.rewardsPlt.showGrid(x=True, y=True)
        self.actionsPlt.showGrid(x=True, y=True)
        
        self.update_plots()
        self.update_statistics()

    def start_timer(self):
        """Start the update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(500)  # Update every 500ms

    def closeEvent(self, event):
        """Handle window close event"""
        if hasattr(self, 'ros_subscriber') and self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    win = Window()
    win.start_timer()

    def shutdown_handler(sig, frame):
        print('DDPG result graph shutdown')
        if hasattr(win, 'ros_subscriber') and win.ros_subscriber is not None:
            win.ros_subscriber.destroy_node()
        rclpy.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()