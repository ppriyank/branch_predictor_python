import math
import os
from random import randint
from itertools import cycle
from typing import List, Tuple, Optional
from abc import abstractmethod
import numpy as np


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


class YehPatt(BranchPredictor):
    def __init__(self, m: int, n: int) -> None:
        super().__init__(n)
        self.n = n
        self.history_table = [0] * 2 ** m
        self.rightmost_m_bits = 2 ** m - 1
        self.nth_bit_from_the_right = 1 << (n-1)

    def get_history_index(self, address: int) -> int:
        """
        The rightmost m bits of the address, not including the final 2 bits which are always 0, are used
        as the index in the history table. 
        """
        return address & self.rightmost_m_bits

    def get_prediction_index(self, address: int) -> int:
        """
        The address determines the index into the history table at which the index into the prediction table is stored.
        """
        return self.history_table[self.get_history_index(address)]

    def update_history(self, address: int, branch_is_taken: bool) -> None:
        """
        Updates the n-bit history table entry by bitshifting in a 1 if the branch was taken, or a 0 if the branch was not taken
        """
        history_index = self.get_history_index(address)
        history = self.history_table[history_index]
        history >>= 1
        if branch_is_taken:
            history |= self.nth_bit_from_the_right
        self.history_table[history_index] = history

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        """
        Make the prediction as normal, but also update the branch history table.
        """
        prediction_is_correct = super().make_prediction(address, branch_is_taken)
        self.update_history(address, branch_is_taken)
        return prediction_is_correct

    @property
    def size(self) -> int:
        return super().size + len(self.history_table) * self.n 

class TageComponent:
    def __init__(self, history_len: int, counter_bits: int = 2) -> None:
        self.tag_len = history_len
        self.counter_bits = counter_bits
        self.history_mask = 2 ** history_len - 1
        self.counter_max = 2 ** counter_bits - 1
        self.threshold = 2 ** (counter_bits - 1)
        self.counter = self.threshold
        self.tag = 0
        self.useful = 0

    def create_tag(self, address: int, history: int) -> int:
        return address ^ (history & self.history_mask)

    def set_tag(self, address: int, history: int) -> None:
        self.tag = self.create_tag(address, history)

    def compare_tag(self, address: int, history: int) -> bool:
        return self.create_tag(address, history) == self.tag

    def decrement_useful(self) -> None:
        if self.useful > 0:
            self.useful -= 1

    def increment_useful(self) -> None:
        if self.useful < 3:
            self.useful += 1

    def update_counter(self, branch_is_taken) -> None:
        if branch_is_taken and self.counter < self.counter_max:
            self.counter += 1

        elif not branch_is_taken and self.counter > 0:
            self.counter -= 1

    def reset_useful(self, reset_msb: bool) -> None:
        if reset_msb:
            self.useful &= 1 
        else:
            self.useful &= 2 

    def reset_component(self, address: int, history: int) -> None:
        self.useful = 0
        self.counter = self.threshold
        self.set_tag(address, history)

    @property
    def prediction(self) -> bool:
        return self.counter >= self.threshold

    @property
    def size(self) -> int:
        return self.tag_len + self.counter_bits + 2 


class Tage:
    def __init__(self, m: int, num_tage_components: int = -1, useful_reset_period: int = 50_000) -> None:
        self.bimodal = Bimodal(m)
        self.tage_components: list[TageComponent] = []

        if num_tage_components == -1:
            log_2_of_m = math.log2(m)
            num_tage_components = math.floor(log_2_of_m)
            if num_tage_components != log_2_of_m:
                num_tage_components += 1
        
        for i in range(1, num_tage_components + 1):
            self.tage_components.append(TageComponent(2**i))

        self.history = 0
        self.history_mask = self.tage_components[-1].history_mask

        self.useful_reset_period = useful_reset_period
        self.useful_reset_countdown = useful_reset_period
        self.reset_msb = True

    def update_history(self, branch_is_taken: bool) -> None:
        self.history <<= 1
        if branch_is_taken:
            self.history += 1
        self.history &= self.history_mask


    def allocate_new_provider(self, address: int, current_provider_index: int = -1) -> None:
        starting_index = current_provider_index + 1
        candidate_new_providers = [component for component in self.tage_components[starting_index:] if component.useful == 0]
        if len(candidate_new_providers) == 0:
            for component in self.tage_components[starting_index:]:
                component.decrement_useful()
            return 

        if len(candidate_new_providers) == 1:
            new_provider = candidate_new_providers[0]
        
        else:
            for candidate in cycle(candidate_new_providers):
                if randint(0, 1) == 1:
                    new_provider = candidate
                    break
        
        new_provider.reset_component(address, self.history)
        if current_provider_index != -1:
            self.tage_components[current_provider_index].useful = 0

    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        bimodal_is_correct = self.bimodal.make_prediction(address, branch_is_taken)
        bimodal_prediction = bimodal_is_correct == branch_is_taken

        provider = None
        provider_index = -1
        alternate = None

        # The provider and the alternate are the two components with the longest histories that have a tag that matches the current address and history
        # If such a compoenent does not exist, the bimodal subs in as a backup
        for i, component in enumerate(self.tage_components[::-1]):
            if component.compare_tag(address, self.history):
                if provider is None:
                    provider = component
                    provider_index = len(self.tage_components) - i - 1
                    continue
                elif alternate is None:
                    alternate = component
                    break

        if provider is None:  # The bimodal is the provider
            # The "provider" was incorrect, so we must allocate a new provider
            self.allocate_new_provider(address)
            return bimodal_is_correct

        provider_prediction = provider.prediction

        if alternate is None:  # The bimodal is the alternate
            alternate_prediction = bimodal_prediction
        else:
            alternate_prediction = alternate.prediction
        
        ### Update the usefulness counters ###
        if provider_prediction != alternate_prediction:
            if provider_prediction == branch_is_taken:
                provider.increment_useful()
            else:
                provider.decrement_useful()
        
        ### Periodically gracefully reset the usefulness counters ###
        self.useful_reset_countdown -= 1
        if self.useful_reset_countdown == 0:
            self.useful_reset_countdown = self.useful_reset_period
            for component in self.tage_components:
                component.reset_useful(self.reset_msb)
            self.reset_msb = not self.reset_msb

        
        # Update the provider's prediction counter, regardless as to whether it was right or wrong
        provider.update_counter(branch_is_taken)

        if provider_prediction == branch_is_taken:
            return True

        # The provider was incorrect, so we must allocate a new provider
        self.allocate_new_provider(address, provider_index)
        return False

    @property
    def size(self) -> int:
        total_component_size = sum([component.size for component in self.tage_components])
        global_history_bits = self.tage_components[-1].tag_len
        return total_component_size + global_history_bits + self.bimodal.size

class PShare:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.global_history = 0
        self.local_history_table = [0] * (2**n)
        self.prediction_table = [[0] * (2**m) for _ in range(2**n)]
        self.table_size = 2**m
        self.gshare_counter = 0
        self.pshare_mask = (1 << n) - 1
        self.pshare_table = ["SN" for _ in range(2**n)]

    def predict(self, address, outcome):
        index = address % self.table_size
        prediction = self.prediction_table[self.local_history_table[address & ((1 << self.n) - 1)]][index]
        pshare_prediction = self.pshare_table[address & self.pshare_mask]
        prediction = max(prediction - 1, 0) if pshare_prediction == "NT" else min(prediction + 1, 3)
        outcome_int = 1 if outcome == "T" else 0
        self.update_counters(address, outcome_int)
        return prediction
    
    def update_counters(self, address, branch_is_taken):
        index = address % self.table_size
        local_history = self.local_history_table[address & ((1 << self.n) - 1)]
        outcome = bin(local_history).count('1') >= self.n//2
        self.prediction_table[local_history][index] = min(self.prediction_table[local_history][index] + 1, 3) if branch_is_taken else max(self.prediction_table[local_history][index] - 1, 0)
        self.local_history_table[address & ((1 << self.n) - 1)] = ((local_history << 1) | int(branch_is_taken)) & ((1 << self.n) - 1)

    def make_prediction(self, address, branch_is_taken):
        index = address % self.table_size
        prediction = self.prediction_table[self.local_history_table[address & ((1 << self.n) - 1)]][index]
        pshare_prediction = self.pshare_table[address & self.pshare_mask]
        prediction = max(prediction - 1, 0) if pshare_prediction == "NT" else min(prediction + 1, 3)
        self.update_counters(address, branch_is_taken)
        return prediction >= 2

class Tournament:
    def __init__(self, m: int, n: int, k: int = 3) -> None:
        self.gshare = GShare(m, n)
        self.pshare = PShare(m, k)
        self.history_table = [0] * (2**n)
        self.tournament_table = [0] * (2**n)

    def make_prediction(self, address, branch_is_taken) -> bool:
        gshare_prediction = self.gshare.predict_taken(address)
        pshare_prediction = self.pshare.make_prediction(address, branch_is_taken)

        history_index = address & ((1 << self.gshare.n) - 1)
        winner = self.tournament_table[history_index]
        if abs(self.gshare.get_counter(address) - self.gshare.threshold) < abs(self.pshare.prediction_table[self.pshare.global_history][address & ((1 << self.pshare.m) - 1)] - (1 << (self.pshare.m - 1))):
            winner = 0
        elif abs(self.gshare.get_counter(address) - self.gshare.threshold) > abs(self.pshare.prediction_table[self.pshare.global_history][address & ((1 << self.pshare.m) - 1)] - (1 << (self.pshare.m - 1))):
            winner = 1

        self.tournament_table[history_index] = winner
        if winner == 0:
            self.gshare.make_prediction(address, branch_is_taken)
        else:
            self.pshare.predict(address, "T" if branch_is_taken else "NT")

        return gshare_prediction if winner == 0 else pshare_prediction
    
    @property
    def size(self) -> int:
        return self.gshare.size + self.pshare.size + len(self.tournament_table)
    
class GShare_ML(GShare):
    def __init__(self, m: int, n: int, method="running_mean", normalization=True) -> None:
        super().__init__(m, n)
        # pip install river 
        # pip install numpy==1.20.3
        from river import (
            compose, linear_model, metrics, preprocessing, forest, 
            ensemble, model_selection, tree, naive_bayes, neighbors, utils)
        self.normalization = normalization
        total_entries = 2 ** m
        self.dim = 22

        if method == "running_mean":
            print("Running Mean!!")
            self.count = [0 for i in range(total_entries) ]
            self.running_mean_vals = [self.threshold for i in range(total_entries)]
            self.make_model_prediction = self.running_mean
        elif method == "running_mean2":
            print("Running Mean 2 !!")
            self.alpha = 0.75
            self.running_mean_vals = [self.threshold for i in range(total_entries)]
            self.make_model_prediction = self.running_mean2
        elif method == "nearest_pattern" or method == "nearest_pattern2":
            print("...Nearest Pattern....")
            del self.counter_bits, self.normalization
            del self.counter_max, self.threshold, self.prediction_table
            self.k = 3
            self.sol = [{} for i in range(total_entries)]
            for pairs in range(2, self.k+1):
                for k in range(pairs+1):
                    poss = ["1" for i in range(k)] + ["0" for i in range(pairs - k)]
                    self.heapPermutation(poss , pairs)
            if method == "nearest_pattern2":
                self.make_model_prediction = self.nearest_pattern2
            else:
                self.centers = [['1' for j in range(self.k)] for i in range(total_entries)]
                self.make_model_prediction = self.nearest_pattern
        elif method == "logistic":
            del self.counter_bits, self.normalization
            del self.counter_max, self.threshold, self.prediction_table
            self.prediction_table = [compose.Pipeline(linear_model.LogisticRegression()) for i in range(total_entries)]
            self.make_model_prediction = self.river_prediction            
        elif method == "logistic2":
            del self.counter_bits, self.normalization
            del self.counter_max, self.threshold, self.prediction_table
            self.prediction_table = compose.Pipeline(linear_model.LogisticRegression())
            self.make_model_prediction = self.river_prediction2            
        elif method == "Perceptron":
            del self.counter_bits, self.normalization
            del self.counter_max, self.threshold, self.prediction_table
            self.prediction_table = [compose.Pipeline(linear_model.Perceptron()) for i in range(total_entries)]
            self.make_model_prediction = self.river_prediction            
        elif method == "Perceptron2":
            del self.counter_bits, self.normalization
            del self.counter_max, self.threshold, self.prediction_table
            self.prediction_table = compose.Pipeline(linear_model.Perceptron())
            self.make_model_prediction = self.river_prediction2            
        elif method == "ALMA":
            del self.counter_bits, self.normalization
            del self.counter_max, self.threshold, self.prediction_table
            self.prediction_table = [compose.Pipeline(linear_model.ALMAClassifier()) for i in range(total_entries)]
            self.make_model_prediction = self.river_prediction            
        elif method == "ALMA2":
            del self.counter_bits, self.normalization
            del self.counter_max, self.threshold, self.prediction_table
            self.prediction_table = compose.Pipeline(compose.Pipeline(linear_model.ALMAClassifier()))
            self.make_model_prediction = self.river_prediction2            
        ######################## doesnt work ########################
        elif method == "HoeffdingAdaptiveTreeClassifier":
            self.prediction_table = [compose.Pipeline(tree.HoeffdingAdaptiveTreeClassifier()) for i in range(total_entries)]
            self.make_model_prediction = self.river_prediction3            
        elif method == "HoeffdingAdaptiveTreeClassifier2":
            self.prediction_table = compose.Pipeline(tree.HoeffdingAdaptiveTreeClassifier())
            self.make_model_prediction = self.river_prediction4            
        elif method == "GaussianNB":
            self.normalization = False
            self.prediction_table = [compose.Pipeline(naive_bayes.GaussianNB()) for i in range(total_entries)]
            self.make_model_prediction = self.river_prediction3            
        elif method == "GaussianNB2":
            self.normalization = False
            self.prediction_table = compose.Pipeline(naive_bayes.GaussianNB())
            self.make_model_prediction = self.river_prediction4            
        ######################## doesnt work ########################

    def heapPermutation(self, a, size):
        if size == 1:
            for i in range(len(self.sol)):
                self.sol[i]["".join(a)] = 0
            return a
        for i in range(size):
            self.heapPermutation(a, size-1)
            if size & 1:
                a[0], a[size-1] = a[size-1], a[0]
            else:
                a[i], a[size-1] = a[size-1], a[i]
    
    def running_mean(self, prediction_index, branch_is_taken, address=None):
        
        counter = self.prediction_table[prediction_index]
        count = self.count[prediction_index]
        
        running_mean_val = self.running_mean_vals[prediction_index]
        prediction = running_mean_val >= self.threshold
        running_mean_val = (counter + count * running_mean_val) / (count + 1)

        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
            
        self.prediction_table[prediction_index] = counter
        self.count[prediction_index] += 1
        self.running_mean_vals[prediction_index] = running_mean_val
        return prediction == branch_is_taken
    
    def running_mean2(self, prediction_index, branch_is_taken, address=None):
        
        counter = self.prediction_table[prediction_index]
        running_mean_val = self.running_mean_vals[prediction_index]
        prediction = running_mean_val >= self.threshold
        running_mean_val = counter * self.alpha + (1-self.alpha) * running_mean_val

        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
            
        self.prediction_table[prediction_index] = counter
        self.running_mean_vals[prediction_index] = running_mean_val

        return prediction == branch_is_taken

    def nearest_pattern(self, prediction_index, branch_is_taken, address=None):
        prediction = 0 
        current_dict = self.sol[prediction_index]
        centers = self.centers[prediction_index]

        for e in range(self.k-1):
            key = "".join(centers[-1-e: ])
            pred = current_dict[key + '1']  >= current_dict[key + '0'] 
            prediction += int(pred)
            self.sol[prediction_index][ key + str(  int(branch_is_taken)  ) ] += 1
        prediction = prediction / (self.k-1) > 0.5
        centers.pop(0)      
        centers.append( str(  int(branch_is_taken)  ) )
        self.centers[prediction_index] = centers
    
        return prediction == branch_is_taken
    
    def nearest_pattern2(self, prediction_index, branch_is_taken, address=None):    
        prediction = 0 
        current_dict = self.sol[prediction_index]
        prediction = True
        centers = bin(self.branch_history)[2:]
        centers = '0' * (self.n - len(centers)) + centers

        for e in range(1,self.k):
            key = centers[:e]
            pred = current_dict[key + '1']  >= current_dict[key + '0'] 
            prediction += int(pred)
            self.sol[prediction_index][ key + str(  int(branch_is_taken)  ) ] += 1
        
        prediction = prediction / (self.k-1) > 0.5
        return prediction == branch_is_taken
    
    def vectorize(self, x):
        record = {}
        for i in range(self.n):
            extracted = (x) & 1
            x = x >> 1
            record[i] = extracted
        return record

    def river_prediction(self, prediction_index, branch_is_taken, address=None):
        model = self.prediction_table[prediction_index]
        record = self.vectorize(self.branch_history)
        # print(record)
        # if self.normalization:
        #     denom = sum(record) ** 0.5 + 1e-6
        #     record = {i: x / denom for i,x in enumerate(record)}
        # else:
        # record = {i: x for i,x in enumerate(record)}

        prediction = model.predict_proba_one(record)
        prediction = prediction[True] > prediction[False]
        model.learn_one(record, branch_is_taken)

        self.prediction_table[prediction_index] = model 

        return prediction == branch_is_taken
    
    def river_prediction2(self, prediction_index, branch_is_taken, address=None):
        
        model = self.prediction_table
        record = self.vectorize(self.branch_history)

        # return addr
        prediction = model.predict_proba_one(record)
        prediction = prediction[True] > prediction[False]
        model.learn_one(record, branch_is_taken)

        self.prediction_table = model 

        return prediction == branch_is_taken
    
    def river_prediction3(self, prediction_index, branch_is_taken, address=None):
        prediction = True
        model = self.prediction_table[prediction_index]
        record = bin(self.branch_history)[2:]
        record = '0' * (self.n - len(record)) + record
        record = [int(x) for x in record] 
        if self.normalization:
            denom = sum(record) ** 0.5 + 1e-6
            record = {i: x / denom for i,x in enumerate(record)}
        else:
            record = {i: x for i,x in enumerate(record)}

        prediction = model.predict_proba_one(record)
        if True in prediction and False in prediction:
            prediction = prediction[True] > prediction[False]
        elif False in prediction:
            prediction = False
        
        # print(record, branch_is_taken)
        model.learn_one(record, branch_is_taken)
        self.prediction_table[prediction_index] = model 
        return prediction == branch_is_taken
    
    def river_prediction4(self, prediction_index, branch_is_taken, address=None):
        prediction = True
        model = self.prediction_table
        record = bin(self.branch_history)[2:]
        record = '0' * (self.n - len(record)) + record
        record = [int(x) for x in record]
        
        if self.normalization:
            denom = sum(record) ** 0.5 + 1e-6
            record = {i: x / denom for i,x in enumerate(record)}
        else:
            record = {i: x for i,x in enumerate(record)}

        prediction = model.predict_proba_one(record)
        if True in prediction and False in prediction:
            prediction = prediction[True] > prediction[False]
        elif False in prediction:
            prediction = False
        
        model.learn_one(record, branch_is_taken)
        self.prediction_table = model 

        return prediction == branch_is_taken
    
    def vectorize2(self, x):
        record = {}
        for i in range(self.dim):
            extracted = (x) & 1
            x = x >> 1
            record[i] = extracted
        return record

    def river_prediction5(self, prediction_index, branch_is_taken, address=None):
        
        model = self.prediction_table
        record = self.vectorize2(address)

        # return addr
        prediction = model.predict_proba_one(record)
        prediction = prediction[True] > prediction[False]
        model.learn_one(record, branch_is_taken)

        self.prediction_table = model 

        return prediction == branch_is_taken
    
    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        prediction_index = self.get_prediction_index(address)
        prediction_is_correct = self.make_model_prediction(prediction_index, branch_is_taken, address)

        self.update_history(branch_is_taken)
        return prediction_is_correct

    @property
    def size(self) -> Optional[int]:
        return None



class S_Clustering(BranchPredictor):
    def __init__(self, m: int, n: int, method=None) -> None:
        self.branch_history = 0
        self.n = n
        if n!=-1:
            self.nth_bit_from_the_right = 1 << (n-1)

        self.m = 2 ** m 
        self.alpha = 0.5
        self.intial_cluster=  0 
        # self.counter = [0 for e in range (m)]
        if method == "skmean2":
            self.dim = 22
            self.make_model_prediction = self.skmeans3
            self.update_history = self.update_history2
            del self.n, self.branch_history
        else:
            self.dim = self.n
            self.make_model_prediction = self.skmeans            

        self.prediction_table = np.array([  [0.0 for i in range(self.dim)] for e in range (self.m) ], dtype=np.float16)
        
        counter_bits = 2
        self.counter_bits = counter_bits
        self.counter_max = 2 ** counter_bits - 1
        self.threshold = 2 ** (counter_bits - 1)
        self.labels = [self.threshold for _ in range(self.m)]

    def update_history(self, branch_is_taken: bool) -> None:
        self.branch_history >>= 1
        if branch_is_taken:
            self.branch_history |= self.nth_bit_from_the_right

    def update_history2(self, branch_is_taken: bool) -> None:
        return 
        
    def skmeans(self, branch_is_taken, address=None):
        record = self.vectorize_dim(self.branch_history)
        distances = abs(record - self.prediction_table).sum(1)
        assigned_cluster = distances.argmin()

        counter = self.labels[assigned_cluster]
        prediction = counter >= self.threshold

        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
        self.labels[assigned_cluster] = counter
        
        self.prediction_table[assigned_cluster] += self.alpha * (record - self.prediction_table[assigned_cluster])        
        # self.counter[assigned_cluster] += 1

        return prediction == branch_is_taken
    
    def skmeans2(self, branch_is_taken, address=None):
        record = self.vectorize_dim(address)

        distances = abs(record - self.prediction_table).sum(1)
        assigned_cluster = distances.argmin()
        
        stats = self.labels[assigned_cluster]
        prediction = stats[True] >= stats[False] 
        self.prediction_table[assigned_cluster] = self.prediction_table[assigned_cluster] + self.alpha * (record - self.prediction_table[assigned_cluster])
        self.labels[assigned_cluster][branch_is_taken] += 1
        
        return prediction == branch_is_taken
    
    def skmeans3(self, branch_is_taken, address=None):
        record = self.vectorize_dim(address)
        distances = abs(record - self.prediction_table).sum(1)
        assigned_cluster = distances.argmin()
        
        counter = self.labels[assigned_cluster]
        prediction = counter >= self.threshold

        if branch_is_taken and counter < self.counter_max:
            counter += 1
        elif not branch_is_taken and counter > 0:
            counter -= 1
        self.labels[assigned_cluster] = counter
        self.prediction_table[assigned_cluster] += self.alpha * (record - self.prediction_table[assigned_cluster])        
        # self.counter[assigned_cluster] += 1

        return prediction == branch_is_taken
    
    def vectorize(self, x):
        record = {}
        for i in range(self.n):
            extracted = (x) & 1
            x = x >> 1
            record[i] = extracted
        return record
    
    def vectorize_dim(self, x):
        record = [0 for i in range(self.dim) ]
        for i in range(self.dim):
            extracted = (x) & 1
            x = x >> 1
            record[i] = extracted
        record = np.array(record)
        return record
 
    def make_prediction(self, address: int, branch_is_taken: bool) -> bool:
        prediction_is_correct = self.make_model_prediction(branch_is_taken, address)
        self.update_history(branch_is_taken)
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
