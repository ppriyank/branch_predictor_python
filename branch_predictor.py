import os
from typing import List, Tuple
from abc import abstractmethod
import numpy as np


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


class Smith(BranchPredictor):
    """
    The simple Smith branch predictor.
    """
    def __init__(self, num_bits) -> None:
        """
        A prediciton table with only 1 element is used to hold our counter so that Smith can share the 
        same basic structure as the other predictors and inherit from the same base class.
        """
        super().__init__(0, counter_bits=num_bits)

    def get_prediction_index(self, address: int) -> int:
        """
        Smith only has one counter, which is at "index 0" in the "prediction table".
        """
        return 0


class Bimodal(BranchPredictor):
    """
    The Bimodal branch predictor.
    """
    def __init__(self, m) -> None:
        """
        These solutions have been formulated such that Bimodal is the "default" and the other predictors are
        modifications of it. Thus, the base class is essentially just a Bimodal predictor as-is.
        """
        super().__init__(m)

    def get_prediction_index(self, address: int) -> int:
        """
        The rightmost m bits of the address, not including the final 2 bits which are always 0, are used
        as the index in the prediction table.
        """
        return (address >> 2) & self.rightmost_m_bits


class GShare(BranchPredictor):
    """
    The GShare branch predictor.
    """
    def __init__(self, m, n) -> None:
        """
        In addition to everything a Bimodal predictor does, the GShare predictor also needs to keep
        track of the most recent n branches.
        """
        super().__init__(m)
        self.branch_history = 0
        self.n = n
        self.nth_bit_from_the_right = 1 << (n-1)

    def get_prediction_index(self, address: int) -> int:
        """
        The rightmost m bits of the address, not including the final 2 bits which are always 0, are XORed
        with the branch history to calculate the index in the prediciton table.
        """
        return ((address >> 2) & self.rightmost_m_bits) ^ self.branch_history

    def make_prediction(self, address: int, branch_is_taken: bool) -> int:
        """
        Make the prediction as normal, but also update the branch history.
        """
        prediction_is_correct = super().make_prediction(address, branch_is_taken)
        self.branch_history >>= 1
        if branch_is_taken:
            self.branch_history |= self.nth_bit_from_the_right
        return prediction_is_correct


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
