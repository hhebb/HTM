
class Cell:
    #segments                   
    # segment list
    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.segments = []            # segment id
        self.connectedSynapses = set()   # temp




class Segment:
    #cellFrom                   # cell id
    #synapses                   # synapse list
    def __init__(self, cell_id, segment_id):
        self.synapses = []                    # synapse object
        self.cell = cell_id
        self.segment_id = segment_id
        self.numActivePotentialSyns = 0
        
    def __del__(self):
        pass


# In[1]:


class Synapse:
    #segmentFrom
    #permanence
    #self.prevSynapticCells     # cell id
    def __init__(self, segment_id, presynaptic, syn_id, permanence):
        self.segment = segment_id           # segment id
        self.preSynapticCell = presynaptic  # cell id
        self.synapse_id = syn_id
        self.permanence = permanence
        
    def __del__(self):
        pass
        


# In[64]:


# 모든 연결 정보 저장
class connectionInfo:
    def __init__(self, Num_TotalCells):
        self.totalCells = [Cell(i) for i in range(Num_TotalCells)] # all cell object
        self.totalSegments = []     # all segment object
        self.segment_order = 0
        self.synapse_order = 0
        
    # segment 생성
    def createNewSegment(self, cell_id):
        newSegment = Segment(cell_id, self.segment_order)
        self.totalCells[cell_id].segments.append(newSegment.segment_id)
        self.totalSegments.append(newSegment)
        self.segment_order += 1
        
        return newSegment.segment_id
        
    # synapse 생성    
    def createNewSynapse(self, segment, preSynaptic, permanence):
        self.totalSegments[segment].synapses.append(Synapse(segment, preSynaptic, self.synapse_order, permanence))
        self.totalCells[preSynaptic].connectedSynapses.add(self.synapse_order)
        self.synapse_order += 1
        
    
    # segment 삭제
    def destroySegment(self, seg_id):
        # seg 인덱스를 받아서 그걸 토대로 삭제
        for seg in self.totalSegments:
            if(seg.segment_id == seg_id):
                cell = seg.cell
                self.totalCells[cell].segments.remove(seg_id) # cell 에서 삭제
                self.clearSynapses(seg_id) # synapses 삭제
                del(seg) # segment 객체 삭제
                #print("segment {} destroyed".format(seg_id))
                break;
            
    
    # synapse 다 날리기
    def clearSynapses(self, seg_id):
        for syn in self.totalSegments[seg_id].synapses:
            self.totalCells[syn.preSynapticCell].connectedSynapses.remove(syn.synapse_id)
            
        # 이후에 segment 삭제되면 알아서 다 삭제됨
        # presynaptic cell 에서만 제거해줌
    
    
    # synapse segment 에서 하나 삭제
    def destroySynapse(self, seg_id, exclude_cells):
        
        min_perm = 1
        min_syn = 0
        # segment 내의 synapse 들의 presynapticCell 이 exclude 에 포함되어있지 않고 permanence 가 가장 작은 것 파괴
        for syn_order in range(len(self.totalSegments[seg_id].synapses)):
            s = self.totalSegments[seg_id].synapses[syn_order]
            if(s.preSynapticCell not in exclude_cells and s.permanence < min_perm):
                min_perm = s.permanence
                min_syn = syn_order
        
        # presynaptic 제거
        target = self.totalSegments[seg_id].synapses[min_syn]
        self.totalCells[target.preSynapticCell].connectedSynapses.remove(target.synapse_id)
        
        # 객체 제거
        del(self.totalSegments[seg_id].synapses[min_syn])

