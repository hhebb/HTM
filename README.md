# HTM :+1: - [original theory from numenta]
Hierachical Temporal Memory System


---
## 1. HTM??

  HTM 은 대뇌 신피질 (neocortex) 의 구조를 벤치마킹한 신경망. 신피질 전역에서 기둥형태의 뉴런집합체인  micro column 구조가 발견된다.
이 구조는 어느곳에서나 같은 구조를 띄고 있으며 어디에 위치하는가에 관계없이 실제로 같은 작업을 수행한다. 
다른 동물들은 감각기관으로부터 받는 정보 대부분을 변연계에서 처리하는 것과 달리 인간은 신피질이 이 작업을 많은 부분 이어받았다.
일반화, 추상화, 추론 등 고등지능은 이 신피질의 계층구조에서 학습되는 듯하다.
HTM 은 이 신피질의 micro column 을 모방한 생체 신경망이다.


## 2. 특징

비지도 학습에 해당.
주로 시계열 데이터를 학습하고 처리.
deep learning 과 다르게 live streaming 데이터를 학습하는데 특화되어있음.
구조가 복잡한 대신 학습은 단순한 hebbian learning 으로 학습.
noise tolerance 가 뛰어남.


## 2. 구조

실제로는 6층의 구조로 되어있지만 1층만 구현해도 sequence data 를 학습할 수 있다고 함.

하나의 계층에 많은 micro column 으로 구성되어 있음.
실제로는 2차원으로 퍼져있지만 1차원으로 구현해도 됨.
하나의 column 안에는 여러개의 cell 이 1열로 기둥형태로 구성되어짐.
하나의 cell 은 2종류의 수상돌기가 있음.
하나는 입력 데이터와 연결되는 proximal segment
다른 하나는 같은 계층내 다른 cell 들과 연결되는 distal segment
proximal segment 는 column 내 cell 들과 모두 공유되므로 이 수상돌기는 column 의 속성으로 간주한다.
이 수상돌기에는 입력 데이터와 직접 연결되는 synapse 들을 갖고 있다.
distal segment 는 각 cell 마다 여러개 갖고 있고 각 segment 마다 여러개의 synapse 를 갖고 있다. 
"olumn - cell - segment - synapse"

크게 2 가지 모듈
1. Spatial Pooler (공간 풀러)
2. Temporal Memory (시간 풀러)

공간 풀러에서는 데이터를 받아서 그것을 특수한 표상을 만들어낸다.
이를 희소분포표상 (SDR) 이라 함.
SDR 은 입력 데이터에 대해 극소수의
