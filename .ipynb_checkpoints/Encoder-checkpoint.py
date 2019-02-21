import numpy as np
import math
import random
import matplotlib.pyplot as plt


class ScalarEncoder:
    def __init__(self, min_val=0, max_val=20, log_max=1000, delta_min=-10, delta_max=10, out_size=16, w=3):
        
        self.min_val = min_val                  # 표현할 최소값
        self.max_val = max_val                  # 표현할 최대값
        self.log_max = log_max               # 로그 encoder 최대값
        self.delta_min = delta_min           # delta 최소값
        self.delta_max = delta_max           # delta 최대값
        self.out_size = out_size             # 출력 데이터 크기
        self.w = w                           # 1 인 bit 의 수
        self.num_bucket = out_size - w + 1   # bucket 의 갯수

    
    
    '''
    가장 기본적인 Encoder.
    제한된 작은 숫자를 인코딩
    ''' 
    def encode_vanilla(self, value):
        
        range_val = self.max_val - self.min_val   # 범위

        # 입력 데이터 제한
        if(value > self.max_val):
            value = self.max_val
        elif(value < self.min_val):
            value = self.min_val
            
        i = math.floor(self.num_bucket * (value - self.min_val) / range_val)
        if(i + self.w >= self.out_size):
            i -= 1
        encoded_data = np.zeros(self.out_size)
        encoded_data[i : i+self.w] = 1

        return encoded_data
    
    
    
    def encode_log(self, value):
        
        range_val = self.log_max - self.min_val   # 범위

        # 입력 데이터 제한
        if(value > self.log_max):
            value = self.log_max
        elif(value < self.min_val):
            value = self.min_val

        i = math.floor(self.num_bucket * np.log10(value - self.min_val) / np.log10(range_val))

        encoded_data = np.zeros(self.out_size)
        encoded_data[i : i+self.w] = 1

        return encoded_data
    
    
    
    def encode_delta(self, value_prev, value_cur):
        delta = value_cur - value_prev
        range_val = self.delta_max - self.delta_min   # 범위

        # 입력 데이터 제한
        if(delta > self.delta_max):
            delta = self.delta_max
        elif(delta < self.delta_min):
            delta = self.delta_min

        i = math.floor(self.num_bucket * (delta - self.delta_min) / range_val)
        encoded_data = np.zeros(self.out_size)
        encoded_data[i : i+self.w] = 1
 
        return encoded_data



    '''
    random hash encode - 기본적으로 사용
    hashing 을 사용해 output data 를 넓게 분포시킨다.
    한정된 output size 일 경우 해상도를 높일 수 있다.
    2 중 hashing 쓰기
    '''
    def encode(self, value):
        
        range_val = self.max_val - self.min_val   # 범위

        # 입력 데이터 제한
        if(value > self.max_val):
            value = self.max_val
        elif(value < self.min_val):
            value = self.min_val

        i = math.floor(self.num_bucket * (value - self.min_val) / range_val)
        if(i + self.w >= self.out_size):
            i -= 1
        encoded_data = np.zeros(self.out_size)
        
        for n in range(i, i+self.w):
            # 2 중 해싱
            key = ((n*n) % (self.out_size) + (n*n) % (3)) % (self.out_size)
            
            # 충돌 처리
            while(encoded_data[key] == 1):
                key += 4
                if(key >= self.out_size):
                    key = 0
                    
            encoded_data[key] = 1
            
        return encoded_data
    

class CategoryEncoder:
    '''
    간단한 카테고리 encoder
    겹치지않게 만든다.
    '''
    def __init__(self, category, w=3):
        self.category = category
        self.category_count = len(category)
        self.active_bits = w
        self.total_bit = self.category_count * self.active_bits
        self.encoded_data = np.zeros(self.total_bit)
    
    def encode(self, item):
        if(item not in self.category):
            print('N/A')
            return None
        
        start_bit = self.category.index(item) + (self.active_bits - 1)*self.category.index(item)
        self.encoded_data[start_bit : start_bit+self.active_bits] = 1
        
        return self.encoded_data



class CycleEncoder:
    '''
    겹치게 만든다.
    case by case 로 hardcoding 이 필요함.
    '''
    def __init__(self, category, active_bits=3, resolution=5):
        self.category = category
        self.category_count = len(category)
        self.active_bits = active_bits
        self.resolution = resolution
        self.total_bit = self.category_count * self.resolution
        self.encoded_data = np.zeros(self.total_bit)
    
    def encode(self, time):
        #start_bit = 
        
        return self.encoded_data



class CategoryEncoder:
    '''
    간단한 카테고리 encoder
    겹치지않게 만든다.
    '''
    def __init__(self, category, active_bits=3):
        self.category = category
        self.category_count = len(category)
        self.active_bits = active_bits
        self.total_bit = self.category_count * self.active_bits
        self.encoded_data = np.zeros(self.total_bit)
    
    def encode(self, item):
        if(item not in self.category):
            print('N/A')
            return None
        
        start_bit = self.category.index(item) + (self.active_bits - 1)*self.category.index(item)
        self.encoded_data[start_bit : start_bit+self.active_bits] = 1
        
        return self.encoded_data



def WeekEncoder(day):
    '''
    주중과 주말을 category 화 하여 encoding.
    - SUN - MON - TUE - WED - THU - FRI - SAT -
    겹치지않게 만든다.
    '''
    data_size = 10
    
    if(day == 'SUN' or day == 'SAT'):
        start_bit = 0
    elif(day == 'MON' or day == 'TUE' or day == 'WED' or 
         day == 'THU' or day == 'FRI'):
        start_bit = int(data_size / 2)
    else:
        print("N/A")
        return 0
        
    encoded_data = np.zeros(data_size)
    encoded_data[start_bit : start_bit + int(data_size/2)] = 1
    
    return encoded_data

