import math
import os
from typing import List, Tuple, Optional
from abc import abstractmethod

class BranchPredictor:
    """
    A base class to handle the functionality that all branch predictors have in common.
    """
    def __init__(self, m: int, counter_bits: int = 3) -> None:
        self.counter_bits = counter_bits
        self.counter_max = 2 ** counter_bits - 1
        self.threshold = 2 ** (counter_bits - 1)
        self.prediction_table = [self.threshold] * 2 ** m
        self.rightmost_m_bits = 2 ** m - 1

    def get_counter(self, address: int = 0) -> int:
        """
        Returns the current value of the counter associated with the given address.
        """
        prediction_index = self.get_prediction_index(address)
        return self.prediction_table[prediction_index]
        
    def predict_taken(self, address: int = 0) -> bool:
        """
        Returns True if the predictor would predict taken for the given address, False if it would predict not taken.
        """
        return self.get_counter(address) >= self.threshold
        
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

    @property
    def size(self) -> Optional[int]:
        return len(self.prediction_table) * self.counter_bits


class Smith(BranchPredictor):
    """
    The simple Smith branch predictor.
    """
    def __init__(self, num_bits: int) -> None:
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
    def __init__(self, m: int) -> None:
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
        return address & self.rightmost_m_bits


class GShare(BranchPredictor):
    """
    The GShare branch predictor.
    """
    def __init__(self, m: int, n: int) -> None:
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
        return (address & self.rightmost_m_bits) ^ self.branch_history

    def update_history(self, branch_is_taken: bool) -> None:
        """
        Updates the n-bit history register by bitshifting in a 1 if the branch was taken, or a 0 if the branch was not taken
        """
        self.branch_history >>= 1
        if branch_is_taken:
            self.branch_history |= self.nth_bit_from_the_right

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        """
        Make the prediction as normal, but also update the branch history.
        """
        prediction_is_correct = super().make_prediction(address, branch_is_taken)
        self.update_history(branch_is_taken)
        return prediction_is_correct

    @property
    def size(self) -> int:
        return super().size + self.n


class Hybrid:
    """
    A Hybrid predictor that combines a GShare predictor with a Bimodal predictor.
    Not technically a subclass of BranchPredictor because it's internal functionality is very different,
    but it implements make_prediction() with the same I/O spec so it can be used as if it were a subclass of BranchPredictor.
    In other words, think of it as "implementing the BranchPredictor interface".
    """
    def __init__(self, k: int, m_gshare: int, n: int, m_bimodal: int) -> None:
        self.chooser_table = [1] * 2 ** k
        self.rightmost_k_bits = 2 ** k - 1
        self.gshare = GShare(m_gshare, n)
        self.bimodal = Bimodal(m_bimodal)

    def choose_gshare(self, address: int) -> None:
        """
        Returns true if the hybrid predictor would use GShare to predict for the given address,
        False if it would use Bimodal.
        """
        chooser_index = address & self.rightmost_k_bits
        chooser = self.chooser_table[chooser_index]
        return chooser >= 2
        
    def update_chooser_table(self, address: int, gshare_is_correct: bool, bimodal_is_correct: bool) -> None:
        """
        Updates the chooser table entry for the given address depending on whether GShare or Bimodal performed better
        """
        if gshare_is_correct == bimodal_is_correct:
            return

        chooser_index = address & self.rightmost_k_bits
        chooser = self.chooser_table[chooser_index]

        if gshare_is_correct and chooser < 3:
            chooser += 1
        elif bimodal_is_correct and chooser > 0:
            chooser -= 1

        self.chooser_table[chooser_index] = chooser

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        """
        Makes a prediction using either the GShare or Bimodal.
        Updates only updates the prediction table of the method that was used.
        Updates the GShare's global history registor either way.
        Updates the chooser table based on which one would have made the better prediction.
        """
        gshare_is_correct = self.gshare.predict_taken(address) == branch_is_taken
        bimodal_is_correct = self.bimodal.predict_taken(address) == branch_is_taken

        if self.choose_gshare(address):
            prediction_is_correct = self.gshare.make_prediction(address, branch_is_taken)
        else:
            prediction_is_correct = self.bimodal.make_prediction(address, branch_is_taken)
            self.gshare.update_history(branch_is_taken)

        self.update_chooser_table(address, gshare_is_correct, bimodal_is_correct)
        return prediction_is_correct

    @property
    def size(self) -> int:
        return self.bimodal.size + self.gshare.size + len(self.chooser_table) * 2

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

    instructions = [(int(line[:6], 16) >> 2, line[7] == 't') for line in lines]
    
    return instructions
        

def run_predictor(predictor: BranchPredictor, filename: str, return_detailed_output: bool = False, instructions: Optional[List[Tuple[int, bool]]] = None) -> Tuple[int, int]:
    """
    Given an instance of a predictor and the filename of a trace file,
    runs all the branch instructions from the trace file through the predictor
    and returns the total number of predictions and the number of incorrect predictions.
    """
    if instructions is None:
        instructions = load_instructions(filename)
    num_predictions = len(instructions)
    num_mispredictions = 0

    if return_detailed_output:
        detailed_output = {
            (False, False): 0,
            (False, True): 0,
            (True, False): 0,
            (True, True): 0
        } 

    for address, branch_is_taken in instructions:
        prediction_is_correct = predictor.make_prediction(address, branch_is_taken)
        if not prediction_is_correct:
            num_mispredictions += 1

        if return_detailed_output:
            actual = branch_is_taken
            predicted = prediction_is_correct == branch_is_taken
            detailed_output[(actual, predicted)] += 1

    if return_detailed_output:
        return num_predictions, num_mispredictions, detailed_output

    return num_predictions, num_mispredictions
