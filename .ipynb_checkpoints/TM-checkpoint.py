import numpy as np
import random
import nbimporter
import memoryStruct


class TemporalMemory:
    state_learn = 1
    state_active = 2
    state_predict = 3

    def __init__(self, sdr_size, cell=5, perm_conn = .5, max_seg=10, threshold=3, learn=True):
        
        self.learn = learn
        self.columnCount = sdr_size                     # SDR 의 크기(column 갯수) 1D array
        self.cellCount = cell                           # column 당 cell 갯수
        self.initialSegments = 5                        # cell 당 초기 segment 갯수
        self.initialSynapses = 5                        # segment 당 초기 synapse 갯수
        self.threshold_active = threshold               # segment 활성화 될 연결된 synapses 갯수
        self.threshold_learn = threshold
        self.initialPerm = np.random.rand()                           # 초기 synapse permanence
        self.conn_perm = perm_conn
        self.perm_inc = .05
        self.perm_dec = .05
        self.predperm_dec = .01
        self.sample_size = 20                           # subsampling 크기
        self.maxSegmentPerCell = max_seg
        self.maxSynapsPerSegment = 10
        #self.maxConnectedSyns = 10                     # 하나의 cell 에 연결되는 최대 synapse 갯수
        
        self.prev_activeSegments = set()               # 이전에 활성화된 Segments 정보 리스트
        self.prev_matchingSegments = set()            # 이전에 match Segments 정보 리스트
        self.activeSegments = set()                    # 활성화된 Segments 정보 리스트
        self.matchingSegments = set()                  # match Segments 정보 리스트
        
        self.prev_activeCells = set()                  # 이전에 활성화된 Cells 리스트
        self.prev_winnerCells = set()                  # 이전에 winner cells 리스트
        self.activeCells = set()                       # 활성화된 Cells 리스트
        self.winnerCells = set()                       # winner cells 리스트
        self.prev_activeColumns = None
        self.activeColumns = None
        self.bursted = []                              # burst 한 columns
        self.predicted = []                            # predicted columns
        self.timestep = 0                              # 특정 timestep
        self.lastUsedIterationForSegment = dict()      # segment 의 가장 최근 활성화 된 시점 저장
        
        # temporal memory 구성하기
        # column = cell id / cellCount
        # cell creation
        totalCellCount = self.columnCount * self.cellCount
        self.connInfo = memoryStruct.connectionInfo(totalCellCount)      # 신경망의 모든 cell 정보
        
        
        
    ''' main method '''
    # activecolumns 는 numpy array = SDR
    def compute(self, activeColumns):
        ## 이전정보와 현재정보 reset ##
        self.prev_activeCells = self.activeCells
        self.prev_winnerCells = self.winnerCells
        self.prev_activeSegments = self.activeSegments
        self.prev_matchingSegments = self.matchingSegments
        self.prev_activeColumns = self.activeColumns
        
        self.activeCells = set()
        self.winnerCells = set()
        self.activeSegments = set()
        self.matchingSegments = set()
        self.activeColumns = activeColumns
        
        self.bursted = set()
        self.predicted = set()
        
        ## main 알고리즘 ##
        
        ## 표상 발현 ##
        for col in range(self.columnCount):
            
            if(activeColumns[col]):
                # active and has at least one predicted segment(cell)
                segmentToCol = self.segmentForColumn(col, self.prev_activeSegments)
                
                if(len(segmentToCol) > 0):
                    self.activatePredictedColumn(col, segmentToCol)
                    
                # active but has no prediced segment(cell)
                else:
                    self.burstColumn(col)
                    self.bursted.add(col)
                
            # inactive and has at least one predicted segment(cell)
            else:
                if(len(self.segmentForColumn(col, self.prev_matchingSegments)) > 0):
                    self.punishPredictedColumn(col)
        
        ## 예측 ##
        for seg in self.connInfo.totalSegments:

            numActiveConnected = 0
            numActivePotential = 0
            
            for s in seg.synapses:
                if(s.preSynapticCell in self.activeCells):
                    if(s.permanence >= self.conn_perm):
                        numActiveConnected += 1
                        
                    if(s.permanence >= 0):
                        numActivePotential += 1
                        
            if(numActiveConnected >= self.threshold_active):
                self.activeSegments.add(seg.segment_id)
                
            if(numActivePotential >= self.threshold_learn):
                self.matchingSegments.add(seg.segment_id)
                
            seg.numActivePotentialSyns = numActivePotential
            
        # predicted columns
        for seg in self.activeSegments:
            pred_cell = self.connInfo.totalSegments[seg].cell
            col = int(pred_cell / self.cellCount)
            self.predicted.add(col)
            
        # 활성화된 segment 마다 현재 시점을 저장
        if(self.learn):
            for s in self.activeSegments:
                self.lastUsedIterationForSegment[s] = self.timestep
            
            self.timestep += 1

            
# ---------------------------------- helper methods ---------------------------------------- #
# ------------------------------------------------------------------------------------------ #


    ''' 예측했던 cell 을 활성화한다 '''
    def activatePredictedColumn(self, col, segments):
        for seg_id in segments:
            seg = self.connInfo.totalSegments[seg_id]
            self.activeCells.add(seg.cell)
            self.winnerCells.add(seg.cell)
            
            if(self.learn):
                for s in self.connInfo.totalSegments[seg_id].synapses:
                    if(s.preSynapticCell in self.prev_activeCells):
                        s.permanence += self.perm_inc
                        s.permanence = min(s.permanence, 1)
                    else:
                        s.permanence -= self.perm_dec
                        s.permanence = max(0, s.permanence)
                        
                newSynapseCount = (self.sample_size - seg.numActivePotentialSyns)
                self.growSynapses(seg_id, newSynapseCount)
                seg.numActivePotentialSyns += newSynapseCount



    ''' 이전에 예측상태에 있던 cell 이 없을 때 해당 column burst '''               
    def burstColumn(self, col):        
        start = col * self.cellCount
        end = (col + 1) * self.cellCount
        
        for c in range(start, end):
            self.activeCells.add(c)
            
        segmentToCol = self.segmentForColumn(col, self.prev_matchingSegments)
        learningSegment = None
        
        if(len(segmentToCol) > 0):
            learningSegment = self.bestMatchingSegment(segmentToCol)
            winnerCell = self.connInfo.totalSegments[learningSegment].cell
        
        else:
            least_usedCell = self.leastUsedCell(col)
            
            if(least_usedCell != None):
                winnerCell = least_usedCell
            
            if(self.learn):
                # createSegment
                learningSegment = self.connInfo.createNewSegment(winnerCell)
                # dict 에 추가
                self.lastUsedIterationForSegment[learningSegment] = self.timestep
                if(len(self.connInfo.totalCells[winnerCell].segments) > self.maxSegmentPerCell):
                    # 가장 적게 활동하는 segment 찾고 삭제.
                    s = sorted(self.lastUsedIterationForSegment.items(), key=lambda x: x[1])
                    destroy_cand = s[0][0]
                    self.lastUsedIterationForSegment.pop(destroy_cand)
                    self.connInfo.destroySegment(destroy_cand) # segment id 구하기, 가장 오래된 segment
                    
                
        self.winnerCells.add(winnerCell)
        
        if(self.learn and learningSegment != None):
            for s in self.connInfo.totalSegments[learningSegment].synapses:
                if(s.preSynapticCell in self.prev_activeCells):
                    s.permanence += min(self.perm_inc, 1)
                else:
                    s.permanence -= max(0, self.perm_dec)
                    
            
            newSynapseCount = (self.sample_size - self.connInfo.totalSegments[learningSegment].numActivePotentialSyns)
            self.growSynapses(learningSegment, newSynapseCount)

            
                       
    def punishPredictedColumn(self, col):
        if(self.learn):
            for seg_id in self.segmentForColumn(col, self.prev_matchingSegments):
                for s in self.connInfo.totalSegments[seg_id].synapses:
                    if(s.preSynapticCell in self.prev_activeCells):
                        s.permanence -= self.predperm_dec
                        
    ''''''''''''
    
    
    
    ''' 특정 column 에 소속된 Segment 반환  '''
    def segmentForColumn(self, col, segments):
        #count = 0
        ret_segments_id = set()
                   
        for seg_id in segments:       
            if(self.connInfo.totalSegments[seg_id].cell / self.cellCount == col):
                #count += 1
                ret_segments_id.add(seg_id)
            
        return ret_segments_id
    
    
    
    def bestMatchingSegment(self, segment):
        bestMatchingSegment = None
        bestScore = -1
        
        for seg_id in segment:
            seg = self.connInfo.totalSegments[seg_id]
            if(seg.numActivePotentialSyns > bestScore):
                bestMatchingSegment = seg_id
                bestScore = seg.numActivePotentialSyns
                
        return bestMatchingSegment
    
    
    
    def leastUsedCell(self, col):
        fewestSegments = 1000
        
        start = col * self.cellCount
        end = (col + 1) * self.cellCount
        
        for c in range(start, end):
            fewestSegments = min(fewestSegments, len(self.connInfo.totalCells[c].segments))
            
        leastUsed = []
        
        for c in range(start, end):
            if(len(self.connInfo.totalCells[c].segments) == fewestSegments):
                leastUsed.append(c)
        
        if(len(leastUsed) > 0):
            chooseRandom = random.randint(0, len(leastUsed) - 1)
            return leastUsed[chooseRandom]
        else:
            return None                   
        
        
        
    def growSynapses(self, seg_id, newSynapseCount):
        
        seg = self.connInfo.totalSegments[seg_id]
        if(len(seg.synapses) > self.maxSynapsPerSegment):
            return

        candidates = list(self.prev_winnerCells.copy())
        
        while(len(candidates) > 0 and newSynapseCount > 0):
            chooseRandom = random.randint(0, len(candidates) - 1)
            presynapticCell = candidates[chooseRandom]
            candidates.remove(presynapticCell)
            
            ## 하나의 cell 에 연결된 synapse 갯수가 꽉 차 있지 않을 때
            #if(len(self.connInfo.totalCells[presynapticCell].connectedSynapses) < self.maxConnectedSyns):
            
            connected = False

            for s in self.connInfo.totalSegments[seg_id].synapses:
                if(s.preSynapticCell == presynapticCell):
                    connected = True

            if(not connected):
                newSynapse = self.connInfo.createNewSynapse(seg_id, presynapticCell, random.random())
                newSynapseCount -= 1

                if(len(self.connInfo.totalSegments[seg_id].synapses) > self.maxSynapsPerSegment):
                    self.connInfo.destroySynapse(seg_id, list(self.winnerCells))
                
        
        
    ''''''''''''        
    
    # burst columns 를 반환
    def column_bursted(self):
        return list(self.bursted)
    
    
    # 예측 columns 을 반환
    def column_predicted(self):
        return list(self.predicted)
    
    # 이전에 활성화 되었던 column 과 이번에 예측한 column 비교
    def evaluate(self):
        self.predicted & self.prev_activeColumns
        
        

