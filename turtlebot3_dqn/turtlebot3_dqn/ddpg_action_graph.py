#!/usr/bin/env python
#################################################################################
# DDPG Action Graph for TurtleBot3 - Continuous Control Visualization
# Based on DQN action_graph with continuous action adaptation
#################################################################################

import signal
import sys
import threading
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QWidget
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Ros2Subscriber(Node):

    def __init__(self, qt_thread):
        super().__init__('ddpg_progress_subscriber')
        self.qt_thread = qt_thread

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/get_action',
            self.get_array_callback,
            10
        )

    def get_array_callback(self, msg):
        data = list(msg.data)
        
        if len(data) >= 2:
            # DDPG에서는 연속 액션 [linear_vel, angular_vel]
            linear_vel = data[0]
            angular_vel = data[1]
            
            # Linear velocity를 percentage로 변환 (0.0 ~ 0.22 -> 0% ~ 100%)
            linear_percentage = min(100, max(0, int((linear_vel / 0.22) * 100)))
            
            # Angular velocity를 percentage로 변환 (-2.84 ~ 2.84 -> 0% ~ 100%)
            # 음수는 왼쪽, 양수는 오른쪽으로 표시
            angular_abs = abs(angular_vel)
            angular_percentage = min(100, max(0, int((angular_abs / 2.84) * 100)))
            
            # 신호 전송
            self.qt_thread.signal_linear_vel.emit(linear_percentage)
            
            if angular_vel < -0.1:  # 왼쪽 회전
                self.qt_thread.signal_left_turn.emit(angular_percentage)
                self.qt_thread.signal_right_turn.emit(0)
            elif angular_vel > 0.1:  # 오른쪽 회전
                self.qt_thread.signal_left_turn.emit(0)
                self.qt_thread.signal_right_turn.emit(angular_percentage)
            else:  # 직진
                self.qt_thread.signal_left_turn.emit(0)
                self.qt_thread.signal_right_turn.emit(0)
            
            # Raw 값들 전송
            self.qt_thread.signal_linear_raw.emit(f"{linear_vel:.3f}")
            self.qt_thread.signal_angular_raw.emit(f"{angular_vel:.3f}")
            
            # Reward 정보 (기존과 동일)
            if len(data) >= 4:
                self.qt_thread.signal_total_reward.emit(str(round(data[2], 2)))
                self.qt_thread.signal_reward.emit(str(round(data[3], 2)))


class Thread(QThread):

    signal_linear_vel = pyqtSignal(int)
    signal_left_turn = pyqtSignal(int)
    signal_right_turn = pyqtSignal(int)
    signal_linear_raw = pyqtSignal(str)
    signal_angular_raw = pyqtSignal(str)
    signal_total_reward = pyqtSignal(str)
    signal_reward = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.node = None

    def run(self):
        self.node = Ros2Subscriber(self)
        rclpy.spin(self.node)
        self.node.destroy_node()


class Form(QWidget):

    def __init__(self, qt_thread):
        super().__init__(flags=Qt.Widget)
        self.qt_thread = qt_thread
        self.setWindowTitle('DDPG Action State - Continuous Control')

        layout = QGridLayout()

        # Linear velocity progress bar
        self.pgsb_linear = QProgressBar()
        self.pgsb_linear.setOrientation(Qt.Vertical)
        self.pgsb_linear.setValue(0)
        self.pgsb_linear.setRange(0, 100)
        self.pgsb_linear.setStyleSheet("QProgressBar::chunk { background-color: green; }")

        # Left turn progress bar
        self.pgsb_left = QProgressBar()
        self.pgsb_left.setOrientation(Qt.Vertical)
        self.pgsb_left.setValue(0)
        self.pgsb_left.setRange(0, 100)
        self.pgsb_left.setStyleSheet("QProgressBar::chunk { background-color: red; }")

        # Right turn progress bar
        self.pgsb_right = QProgressBar()
        self.pgsb_right.setOrientation(Qt.Vertical)
        self.pgsb_right.setValue(0)
        self.pgsb_right.setRange(0, 100)
        self.pgsb_right.setStyleSheet("QProgressBar::chunk { background-color: blue; }")

        # Raw values display
        self.label_linear_raw = QLabel('Linear Vel (m/s)')
        self.edit_linear_raw = QLineEdit('0.000')
        self.edit_linear_raw.setDisabled(True)
        self.edit_linear_raw.setFixedWidth(100)

        self.label_angular_raw = QLabel('Angular Vel (rad/s)')
        self.edit_angular_raw = QLineEdit('0.000')
        self.edit_angular_raw.setDisabled(True)
        self.edit_angular_raw.setFixedWidth(100)

        # Reward display
        self.label_total_reward = QLabel('Total reward')
        self.edit_total_reward = QLineEdit('')
        self.edit_total_reward.setDisabled(True)
        self.edit_total_reward.setFixedWidth(100)

        self.label_reward = QLabel('Reward')
        self.edit_reward = QLineEdit('')
        self.edit_reward.setDisabled(True)
        self.edit_reward.setFixedWidth(100)

        # Action labels
        self.label_left = QLabel('Left Turn')
        self.label_linear = QLabel('Linear Vel')
        self.label_right = QLabel('Right Turn')

        # Layout arrangement
        # Raw values section
        layout.addWidget(self.label_linear_raw, 0, 0)
        layout.addWidget(self.edit_linear_raw, 1, 0)
        layout.addWidget(self.label_angular_raw, 2, 0)
        layout.addWidget(self.edit_angular_raw, 3, 0)

        # Reward section
        layout.addWidget(self.label_total_reward, 0, 1)
        layout.addWidget(self.edit_total_reward, 1, 1)
        layout.addWidget(self.label_reward, 2, 1)
        layout.addWidget(self.edit_reward, 3, 1)

        # Progress bars
        layout.addWidget(self.pgsb_left, 0, 4, 4, 1)
        layout.addWidget(self.pgsb_linear, 0, 5, 4, 1)
        layout.addWidget(self.pgsb_right, 0, 6, 4, 1)

        # Progress bar labels
        layout.addWidget(self.label_left, 4, 4)
        layout.addWidget(self.label_linear, 4, 5)
        layout.addWidget(self.label_right, 4, 6)

        # Additional info labels
        info_label_1 = QLabel('Green: Forward Speed')
        info_label_2 = QLabel('Red: Left Turn')
        info_label_3 = QLabel('Blue: Right Turn')
        
        layout.addWidget(info_label_1, 5, 4)
        layout.addWidget(info_label_2, 5, 5)
        layout.addWidget(info_label_3, 5, 6)

        self.setLayout(layout)

        # Connect signals
        qt_thread.signal_linear_vel.connect(self.pgsb_linear.setValue)
        qt_thread.signal_left_turn.connect(self.pgsb_left.setValue)
        qt_thread.signal_right_turn.connect(self.pgsb_right.setValue)
        qt_thread.signal_linear_raw.connect(self.edit_linear_raw.setText)
        qt_thread.signal_angular_raw.connect(self.edit_angular_raw.setText)
        qt_thread.signal_total_reward.connect(self.edit_total_reward.setText)
        qt_thread.signal_reward.connect(self.edit_reward.setText)

    def closeEvent(self, event):
        if hasattr(self.qt_thread, 'node') and self.qt_thread.node is not None:
            self.qt_thread.node.destroy_node()
        rclpy.shutdown()
        event.accept()


def run_qt_app(qt_thread):
    app = QApplication(sys.argv)
    form = Form(qt_thread)
    form.show()
    app.exec_()


def main():
    rclpy.init()
    qt_thread = Thread()
    qt_thread.start()
    qt_gui_thread = threading.Thread(target=run_qt_app, args=(qt_thread,), daemon=True)
    qt_gui_thread.start()

    def shutdown_handler(sig, frame):
        print('DDPG action graph shutdown')
        if hasattr(qt_thread, 'node') and qt_thread.node is not None:
            qt_thread.node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == '__main__':
    main()