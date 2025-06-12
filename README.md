## 설치 가이드    
#필요한 패키지  
turtlebot3: SLAM과 Navigation 등 터틀봇에 필요한 패키지를 모두 담고 있다.  
turtlebot3_msgs: DQN에 필요한 state, action, reward 메시지(service message)를 담고 있다.  
turtlebot3_simulations: 심층강화학습을 실행할 각 단계별 Gazebo 시뮬레이션을 담고 있다.  
turtlebot3_machine_learning: 터틀봇에 적용할 수 있는 DQN 코드를 담고 있다.  

## 개선 사항  
코드: dqn_gent.py  
함수: train_model  
1. 학습 방식 변경  
기존: For 루프로 샘플 하나씩 처리  
개선: 배치 전체를 벡터 연산으로 한번에 처리  
2. Q-값 처리 효율화  
기존: 전체 Q-값 벡터 [2.1, 3.5, 1.8, 4.2, 2.9] 저장/계산  
개선: 선택된 Q-값만 3.5 추출/계산  
3. 손실 함수 최적화  
기존: model.fit() - 5개 액션 모두 계산 (4개는 의미없음)  
개선: GradientTape - 선택된 1개 액션만 계산  
4. 성능 향상  

메모리: 60% 적은 메모리 사용  
속도: 불필요한 계산 제거로 더 빠르게 학습  

## 영상  
https://youtu.be/5iGAN8dJ6Io  
