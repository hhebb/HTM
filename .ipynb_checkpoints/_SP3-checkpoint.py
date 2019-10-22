import numpy as np
import collections
import matplotlib.pyplot as plt
import math
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot
import nbimporter
import Encoder
import random
import time


class SpatialPooler:
    def __init__(self, input_size, columns=512, perm_conn=.7, minOver=5, potential_rate=.8):
        self.input_size = input_size                               # input vector 크기
        self.input_data = np.empty([self.input_size])              # encoding 된 input vector
        self.columnCount = columns                                 # column 의 크기 - 1d array
        self.connectedPerm = perm_conn                               # synapse 활성화(1) 될 permanence 임계치
        self.min_overlap = minOver                                 # 발화 하기 위한 컬럼 당 최소한의 overlap count
        self.minGlobalActivity = 0                                 # 승리하기 위해 필요한 score (global inhibition)
        self.desiredGlobalActivity = int(0.02 * self.columnCount)  # 한 번에 승리할 column 수 (global inhibition), 2 % 만 발화
        self.minDutyCycle = 0                                      # column 당 최소 발화 duty
        self.highDutyCycle = 0
        self.permanence_inc = .05                                  # 학습시 permanence 증가량
        self.permanence_dec = .05                                  # 학습시 permanence 감소량
        self.history_capacity = 200                                # duty 계산할 history 용량
        self.step = 0                                              # 데이터 처리한 수
        self.potential_rate = potential_rate                       # 입력 데이터에 대한 potential synapses 의 비율
        self.potential_count = int(self.input_size*self.potential_rate)    # potiential synapses 갯수. 이외의 입력값에는 전혀 연결성이 없다.
        
        self.potential_synapses = np.random.random([self.input_size, self.columnCount])   # potential synapses(proxiaml dendrite) - permanence ndarry. 초기화 필요      
        self.connected_synapses = np.zeros([self.input_size, self.columnCount])           # permanence 가 높아 연결된 synapses
        self.boosts = np.ones([self.columnCount])                                         # 보정에 필요한 boost factor
        self.overlapped = np.zeros([self.columnCount])                                    # input 과 연결된 synapse 들과의 최초 계산
        self.activeColumns = np.zeros([self.columnCount])                                 # 활성화된 columns
        self.activeHistory = []                                            # active duty 를 계산하기 위한 active 기록
        self.overlapHistory = []                                           # overlap duty 를 계산하기 위한 overlap 기록
        self.activeDutyInfo = np.zeros([self.columnCount])                 # active duty 정보
        self.overlapDutyInfo = np.zeros([self.columnCount])                # overlap duty 정보
        self.sparsity_history = []
        
        # potential synapses 구성
        for col in range(self.columnCount):
            never_connect = self.input_size - self.potential_count

            while(never_connect > 0):
                idx = random.randint(0, self.input_size - 1)

                if(self.potential_synapses[idx, col] != -1):
                    self.potential_synapses[idx, col] = -1
                    never_connect -= 1
        
        ## duty 계산을 위한 history 생성 ##
        for c in range(self.columnCount):
            self.activeHistory.append(collections.deque())
            self.overlapHistory.append(collections.deque())
            
                
    ''' SDR 생성 '''
    def compute_SDR(self, input_data, learn=True, debug=False):
        
        self.input_data = input_data
        
        ## 1. overlaping ##
        self.connected_synapses = self.potential_synapses > self.connectedPerm
        self.overlapped = self.input_data @ self.connected_synapses
        
        for c in range(self.columnCount):
            if(self.overlapped[c] > self.min_overlap):
                self.overlapped[c] *= self.boosts[c]

                if(len(self.overlapHistory[c]) >= self.history_capacity):
                    self.overlapHistory[c].popleft()

                self.overlapHistory[c].append(True)

            else:
                self.overlapped[c] = 0

                if(len(self.overlapHistory[c]) >= self.history_capacity):
                    self.overlapHistory[c].popleft()

                self.overlapHistory[c].append(False)
                    
                    
        ## 2. inhibition (global) ##
        self.minGlobalActivity = self.kthScore(self.desiredGlobalActivity)
        self.activeColumns = self.overlapped >= self.minGlobalActivity
        
        for c in range(self.columnCount):                
            if(len(self.activeHistory[c]) >= self.history_capacity):
                self.activeHistory[c].popleft()

            self.activeHistory[c].append(self.activeColumns[c])
                    
                
        ## 3. learning ## 
        if(learn):
            for c in range(self.columnCount):

                if(self.activeColumns[c] == 1):
                    for s in range(self.input_size):

                        if(self.potential_synapses[s, c] != -1):
                            if(self.input_data[s] == 1):
                                self.potential_synapses[s, c] += self.permanence_inc
                                self.potential_synapses[s, c] = min(self.potential_synapses[s, c], 1.0)
                            else:
                                self.potential_synapses[s, c] -= self.permanence_dec
                                self.potential_synapses[s, c] = max(0.0, self.potential_synapses[s, c])


        self.update_activeDuty()
        self.update_overlapDuty()
        
        self.step += 1
        
        if(debug):
            sparsity = (np.count_nonzero(self.activeColumns==True)/(self.columnCount))
            self.sparsity_history.append(sparsity)
        
        
        ## 3.2. 보정 작업 ##
        if(learn):
            for c in range(self.columnCount):
                ## 각 active duty 에 따라 매번 boost 시켜줌
                maxDuty, highDuty = self.maxhighDutyCycle()
                self.minDutyCycle = .01 * maxDuty
                #self.highDutyCycle = highDuty
                self.boostFunction(c)

                ## input 과 잘 겹치지 않는 synapse 에 대해서 permanence 증가시켜줌
                if(self.overlapDutyInfo[c] < self.minDutyCycle):
                    self.increase_Permanence(c)

                
# ----------------------------------- helper method ---------------------------------------- #
# ------------------------------------------------------------------------------------------ #

    ''' global 하게 승리할 컬럼의 기준 '''
    def kthScore(self, desired_kth):
        
        rank = self.overlapped.ravel().copy()
        rank.sort()        
        score = rank[-desired_kth]
        
        return score
    
    
    ''' global 하게 가장 자주 승리한 컬럼의 duty '''
    def maxhighDutyCycle(self):
        
        rank = self.activeDutyInfo.ravel().copy()
        rank.sort()
        maxDuty = rank[-1]
        highDuty = rank[-int(self.input_size/5)]
        
        return maxDuty, highDuty
    
    
    
    ''' 해당 column 이 발화하도록 격려 '''
    def boostFunction(self, c):
        
        x = self.activeDutyInfo[c]
        
        if(x > .02):
            self.boosts[c] = -5*x + 1.1
        else:
            self.boosts[c] = -75*x + 2.5
            
            
            
    ''' 해당 column 의 모든 petential synapse 의 permanence 를 증가시켜 잘 겹치도록 격려 '''
    def increase_Permanence(self, c):
        for s in range(self.input_size):
            if(self.potential_synapses[s, c] != -1):
                self.potential_synapses[s, c] += self.permanence_inc
        
        
    ''' activeDuty update '''
    def update_activeDuty(self):
        for c in range(self.columnCount):
            self.activeDutyInfo[c] = np.mean(self.activeHistory[c])

                
    ''' overlapDuty update '''
    def update_overlapDuty(self):
        for c in range(self.columnCount):
            self.overlapDutyInfo[c] = np.mean(self.overlapHistory[c])
    
    
    
    
# ----------------------------------- Test & Visualization Methods ---------------------------------------- #
# ------------------------------------------------------------------------------------------ #
    
    ''' Testing & Visualization '''
    # 검증 tool 스크립트로 옮길 예정
    
    
    ''' active columns 반환 (raw numpy array) '''
    def getActiveColumns_raw(self):
        return self.activeColumns
    
    ''' active columns 반환 (active column index)'''
    def getActiveColumns(self):        
        return np.asarray(np.where(self.activeColumns == 1))[0]
    
    
    ''' 두 SDR 의 유사도 반환 & 시각화 '''
    # legend 추가, hover 추가
    def similarity(self, SDR_1, SDR_2, viz=True):
        
        if(viz):
            '''
            N = self.columnCount
            img_1 = np.zeros([1, N, 4], dtype=np.uint8)
            img_2 = np.zeros([1, N, 4], dtype=np.uint8)

            output_notebook()

            for i in range(N):
                if(SDR_1[i]):
                    img_1[0, i, 0:3] = 0 # r

                else:
                    img_1[0, i, 0:3] = 255 # r


                img_1[0, i, 3] = 255 # a

            for i in range(N):
                if(SDR_2[i]):
                    img_2[0, i, 0:3] = 0 # r

                else:
                    img_2[0, i, 0:3] = 255 # r


                img_2[0, i, 3] = 255 # a

                
            p = figure(plot_width=1000, plot_height=100, x_range=(0, self.columnCount), y_range=(0, 1))
            p2 = figure(plot_width=1000, plot_height=100, x_range=p.x_range, y_range=p.y_range)

            p.image_rgba(image=[img_1], x=[0], y=[0], dw=[self.columnCount], dh=[1])
            p2.image_rgba(image=[img_2], x=[0], y=[0], dw=[self.columnCount], dh=[1])

            show(gridplot([[p],[p2]]))
            '''
            
            output_notebook()
            p = figure(plot_width=self.columnCount, plot_height=100, x_range=(0, self.columnCount))
            p2 = figure(plot_width=self.columnCount, plot_height=100, x_range=p.x_range)
            p.rect(x=SDR_1, y=0, width=.9, height=.9)
            p2.rect(x=SDR_2, y=0, width=.9, height=.9)
            
            p.grid.grid_line_color = None
            p.axis.axis_line_color = None
            p.axis.major_tick_line_color = None
            p.axis.major_label_standoff = 0
            
            p2.grid.grid_line_color = None
            p2.axis.axis_line_color = None
            p2.axis.major_tick_line_color = None
            p2.axis.major_label_standoff = 0
            
            show(gridplot([[p],[p2]]))
             
        
        # 유사도 = 1 이면, 두 sdr 완벽히 일치
        print("유사도 :", len(np.intersect1d(SDR_1, SDR_2)) / len(SDR_1))
        print(SDR_1)
        print(SDR_2)

        
        
    ''' 같은 입력에 대해서 같은 출력을 내는 정도  '''
    def consistency(self, data, num):
        
        mismatch = 0
        sdr = None
        
        self.compute_SDR(data, False)
        sdr_prev = self.getActiveColumns()
        
        for i in range(num):
            self.compute_SDR(data, False)
            sdr = self.getActiveColumns()
            mismatch += np.count_nonzero(sdr ^ sdr_prev)
            sdr_prev = sdr.copy()
            
        # 오류 0 이면 모든 테스트케이스에 대하여 동일한 출력.
        print("오류 :", mismatch / (self.desiredGlobalActivity*num))
            
            
    
    def noiseRobustness(self):
        # noise 에 대해 견고성 테스트 결과
        pass
    
    def falseRobustness(self):
        # 오류에 대해 견고성 테스트 결과
        pass
    
    
                        
    ''' active columns(SDRs), overlap 카운트  '''                    
    def viz_activeCol(self):
        
        output_notebook()
        
        N = self.columnCount
        img = np.zeros([1, N, 4], dtype=np.uint8)
        
        maxOverlap = np.sort(self.overlapped).copy()[-1]
        
        for i in range(N):
            if(self.activeColumns[i]):
                img[0, i, 0:3] = 0 # r

            else:
                img[0, i, 0:3] = 255 # r
                
            
            img[0, i, 3] = np.sqrt(self.overlapped[i]/maxOverlap) * 255 # a
        # 
        p = figure(plot_width=1000, plot_height=100)
        p.image_rgba(image=[img], x=[0], y=[0], dw=[self.columnCount], dh=[1])
        show(p)
        
        idx = []
        for i in range((self.columnCount)):
            if(self.activeColumns[i] == 1):
                idx.append(i)
        print("active columns : {} \ncount : {}".format(idx, np.count_nonzero(self.activeColumns)))
        
    
    ''' 현재 신경망 상태들 '''
    def viz_NetState(self):
        # SDR 의 현재 상태
        # active duty, overlap duty, sparsity 평균, 표준편차
        # permanenece, boost factor
        output_notebook()
        
        print('> desired active column count : {} <\n'.format(self.desiredGlobalActivity))

        p = figure(plot_width=800, plot_height=300, title="Boost Factor", x_axis_label='column number', y_axis_label='boost factor', y_range=(0,3))
        p.line(np.arange(self.columnCount), self.boosts, legend="factor", line_width=.5)
        show(p)
        print('boost factor\n')
        print("-> 평균 : {}, 표준편차 : {}\n".format(self.boosts.mean(), self.boosts.std()))
        
        p = figure(plot_width=800, plot_height=300, title="active duty", x_axis_label='column number', y_axis_label='duty', y_range=(0,.5))
        p.line(np.arange(self.columnCount), self.activeDutyInfo, legend="factor", line_width=.5)
        show(p)
        print('active duty\n')
        print("-> 평균 : {}, 표준편차 : {}\n".format(self.activeDutyInfo.mean(), self.activeDutyInfo.std()))
        
        p = figure(plot_width=800, plot_height=300, title="overlap duty", x_axis_label='column number', y_axis_label='duty', y_range=(0,1))
        p.line(np.arange(self.columnCount), self.overlapDutyInfo, legend="factor", line_width=.5)
        show(p)        
        print('overlap duty\n')
        print("-> 평균 : {}, 표준편차 : {}\n".format(self.overlapDutyInfo.mean(), self.overlapDutyInfo.std()))
        
    
    
    def test(self):
        idx = []
        for i in range(len(self.activeColumns)):
            if(self.activeColumns[i] == 1):
                idx.append(i)
                
        return idx
    
    
    ''' sparsity 시각화 출력 - 변화 '''
    def viz_sparsity_period(self):
        #fig = plt.figure(figsize=(20,1))
        #plt.imshow(self.activeColumns.reshape(1, self.columnCount))
        #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        #cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        #plt.colorbar(cax=cax)
        #plt.show()
        output_notebook()
        
        p = figure(plot_width=800, plot_height=300, title="Sparsity", x_axis_label='step', y_axis_label='sparsity', y_range=(0,.1))
        p.line(np.arange(self.step), self.sparsity_history, legend="sparsity", line_width=.5)
        show(p)
        
        
    ''' active duty 평균 - 변화 '''
    def viz_activeDuty_period(self):
        #fig = plt.figure(figsize=(20,1))
        #plt.imshow(self.activeColumns.reshape(1, self.columnCount))
        #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        #cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        #plt.colorbar(cax=cax)
        #plt.show()
        sparsity = (np.count_nonzero(self.activeColumns==True)/(self.columnCount))
        print('sparsity : {}'.format(sparsity))

        
    def viz_connected_synapses(self, col):
        print(np.count_nonzero(self.connected_synapses[:,col]))
