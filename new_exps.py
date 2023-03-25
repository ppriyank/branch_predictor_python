import os
from typing import List, Tuple
from abc import abstractmethod
import numpy as np
import sys
import random 
from sklearn.cluster import Birch
random.seed(1)

class BranchPredictor:
    """
    A base class to handle the functionality that all branch predictors have in common.
    """
    def __init__(self, m, counter_bits=3) -> None:
        self.counter_max = 2 ** counter_bits - 1
        self.threshold = 2 ** (counter_bits - 1)
        self.prediction_table = np.full(2 ** m, self.threshold)
        self.m = m
        self.rightmost_m_bits = 2 ** m - 1
        
    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        """
        Makes a prediction based on the counter.
        Updates the counter based on if the branch is really taken or not.
        Returns True if the prediction is correct, false otherwise.
        """
        prediction_index = self.get_prediction_index(address)
        counter = self.prediction_table[prediction_index]
        prediction = counter >= self.threshold

        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
            
        self.prediction_table[prediction_index] = counter

        return prediction == branch_is_taken

    @abstractmethod
    def get_prediction_index(self, address: int) -> int:
        pass


class running_mean(BranchPredictor):
    """
    N M = x
	(N+1) M' = x + K =>  M' = (K + NM) / (N+1)
    """
    def __init__(self, num_bits) -> None:
        super().__init__(0, counter_bits=num_bits)
        self.threshold = self.counter_max / 2
        self.running_mean = self.threshold
        self.count = 0 
        print(self.threshold, self.counter_max)

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        prediction_index = self.get_prediction_index(address)
        counter = self.prediction_table[prediction_index]
        prediction = self.running_mean >= self.threshold

        self.running_mean= (counter + self.count * self.running_mean) / (self.count + 1)
        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
        self.prediction_table[prediction_index] = counter
        self.count += 1


        return prediction == branch_is_taken

    def get_prediction_index(self, address: int) -> int:
        return 0

    def get_counter(self):
        return self.prediction_table[0]

class running_mean2(BranchPredictor):
    """
    N M = x
	(N+1) M' = x + K =>  M' = (K + NM) / (N+1)
    """
    def __init__(self, num_bits, alpha=0.75) -> None:
        super().__init__(0, counter_bits=num_bits)
        self.threshold = self.counter_max / 2
        self.running_mean = self.threshold
        self.count = 0 
        self.alpha = alpha
        print(self.threshold, self.counter_max)

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        prediction_index = self.get_prediction_index(address)
        counter = self.prediction_table[prediction_index]
        prediction = self.running_mean >= self.threshold

        self.running_mean= counter * self.alpha  + (1-self.alpha) * self.running_mean

        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
        self.prediction_table[prediction_index] = counter
        self.count += 1

        
        return prediction == branch_is_taken

    def get_prediction_index(self, address: int) -> int:
        return 0

    def get_counter(self):
        return self.prediction_table[0]

class Ensemble(BranchPredictor):
    def __init__(self, num_bits=3) -> None:
        
        self.starting_bits = 1
        if len(num_bits) > 1:
        	self.starting_bits = int(num_bits[1])
        num_bits = int(num_bits[0])
        
        
        self.counter_max = [(2 ** i - 1) for i in range(self.starting_bits, num_bits+1)]
        self.threshold = [2 ** (i - 1) for i in range(self.starting_bits, num_bits + 1)]
        self.prediction_table = [0 for i in range(self.starting_bits, num_bits+1)]

        self.num_bits = num_bits 

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
    	votes = [] 
    	for i in range(self.num_bits - self.starting_bits + 1):
    		counter = self.prediction_table[i]
    		votes.append( float(counter >= self.threshold[i]) )
    		if branch_is_taken and counter < self.counter_max[i]:
    			counter += 1
    		elif not branch_is_taken and counter > 0:
    			counter -= 1
    		self.prediction_table[i] = counter

    	
    	prdiction = (sum(votes) / len(self.prediction_table)) > 0.5
    	return prdiction == branch_is_taken


    def get_prediction_index(self, address: int) -> int:
        """
        Smith only has one counter, which is at "index 0" in the "prediction table".
        """
        return 0

    def get_counter(self):
        return self.prediction_table[0]

class RS(BranchPredictor):
    def __init__(self, num_bits) -> None:
    	num_bits = int(num_bits[0])
    	self.num_bits = num_bits
    	self.predictions = [False for i in range(num_bits)]
    	self.count = 0 
    
    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
    	self.count += 1
    	prediction_index = self.get_prediction_index(address)
    	if prediction_index < self.num_bits :
    		prediction = self.predictions[prediction_index]
    		self.predictions[prediction_index] = branch_is_taken
    	else:
    		# prediction = False
    		prediction = True
    	
    	return prediction == branch_is_taken

    def get_prediction_index(self, address: int) -> int:
    	chosen = random.randint(0, self.count) 
    	# print(chosen)
    	return chosen

    def get_counter(self):
    	chosen = random.randint(0, self.count)
    	if chosen < self.num_bits :
    		prediction = self.predictions[prediction_index]
    	else:
    		prediction = False
    	return prediction

class Random_Bimodal(BranchPredictor):
    def __init__(self, m) -> None:
    	m = int(m[0])
    	super().__init__(m)

    def get_prediction_index(self, address: int) -> int:
    	addr = bin(address >> 2)
    	addr = random.choices(addr[2:], k=self.m)
    	addr = "0b" + "".join(addr)
    	return int(addr,2)

    def get_counter(self):
        return self.prediction_table[0]

class birch(BranchPredictor):
    """
    The Bimodal branch predictor.
    """
    def __init__(self, m) -> None:
        """
        These solutions have been formulated such that Bimodal is the "default" and the other predictors are
        modifications of it. Thus, the base class is essentially just a Bimodal predictor as-is.
        """
        super().__init__(m)

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        prediction_index = self.get_prediction_index(address)
        counter = self.prediction_table[prediction_index]
        prediction = self.running_mean >= self.threshold

        self.running_mean= (counter + self.count * self.running_mean) / (self.count + 1)
        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
        self.prediction_table[prediction_index] = counter
        self.count += 1


        return prediction == branch_is_taken

    def get_prediction_index(self, address: int) -> int:
        """
        The rightmost m bits of the address, not including the final 2 bits which are always 0, are used
        as the index in the prediction table.
        """
        return (address >> 2) & self.rightmost_m_bits



def load_instructions(filename: str) -> List[Tuple[int, bool]]:
    """
    Loads the branch instructions from the trace file.
    Returns a list of tuples of the form (address, branch_is_taken), 
    where address is an integer and branch_is_taken is a bool.
    """
    if not os.path.isfile(filename):
        file_with_parent = os.path.join('traces/', filename)
        if not os.path.isfile(file_with_parent):
            raise FileNotFoundError(f"Trace file '{filename}' not found")
        filename = file_with_parent

    with open(filename, 'r') as f:
        lines = f.readlines()

    instructions = [(int(line[:6], 16), line[7] == 't') for line in lines]
    
    return instructions
        

def run_predictor(predictor: BranchPredictor, filename: str) -> Tuple[int, int]:
    """
    Given an instance of a predictor and the filename of a trace file,
    runs all the branch instructions from the trace file through the predictor
    and returns the total number of predictions and the number of incorrect predictions.
    """
    instructions = load_instructions(filename)
    num_predictions = len(instructions)
    num_mispredictions = 0

    for address, branch_is_taken in instructions:
        prediction_is_correct = predictor.make_prediction(address, branch_is_taken)
        if not prediction_is_correct:
            num_mispredictions += 1
    
    return num_predictions, num_mispredictions




if __name__ == "__main__":
	trace_file = sys.argv[2]
	function_name = sys.argv[1]
	counter_bits = sys.argv[3:]
	predictor = locals()[function_name](counter_bits)

	
	num_predictions, num_mispredictions = run_predictor(predictor, trace_file)

	misprediction_rate = 100 * num_mispredictions / num_predictions

	print(f"number of predictions:		{num_predictions}")
	print(f"number of mispredictions:	{num_mispredictions}")
	print(f"misprediction rate:		{misprediction_rate:.2f}%")
	print(f"FINAL COUNTER CONTENT:		{predictor.get_counter()}")
