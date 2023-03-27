import argparse
from mmcv import DictAction
from typing import List, Tuple
import os 
from sklearn.cluster import MiniBatchKMeans
import numpy as np 
import random 

from MicroCluster import DenStream, DenStream2, DenStream3
random.seed(1)


def get_args_parser():
    parser = argparse.ArgumentParser('ML Streaming', add_help=False)
    parser.add_argument('--algorithm_name', default="---", type=str)
    parser.add_argument('--trace_file', default='', help='trace file')
    parser.add_argument(
        '--additional_args', nargs='+', action=DictAction, default={}, 
        help=' dict argument ' ' this is so fun ' " ... ")
    return parser


def vectorize(x):
	addr = bin(x)[2:]
	addr = np.array([int(e) for e in addr])
	addr = addr / np.linalg.norm(addr)
	addr = addr.reshape(1,-1)
	return addr

class S_Kmeans():
	# https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm
	def __init__(self, n_cluster, dim=22):
		
		self.centers = np.array([[random.choice([0,1]) for i in range(dim) ] for e in range (n_cluster)])
		self.centers = self.centers / np.linalg.norm(self.centers, axis=-1).reshape(-1,1)

		self.labels = [{False: 0, True: 0} for i in range(n_cluster)]
		self.default_pred = False
		self.n_cluster = n_cluster
		# print("Clusters, ", self.centers)

	def make_prediction(self, address, branch_is_taken):

		address = vectorize(address)
		prediction = self.default_pred

		distances = np.linalg.norm(address - self.centers, ord=2, axis=1.) 
		assigned_cluster = np.argmin(distances)

		stats = self.labels[assigned_cluster]
		
		prediction = stats[True] >= stats[False] 

		self.labels[assigned_cluster][branch_is_taken] += 1
		N = sum([key[1] for key in stats.items()])
		
		temp_center = self.centers[assigned_cluster]
		temp_center = temp_center + (1/ N) * (address - temp_center)
		temp_center = temp_center / np.linalg.norm(temp_center, axis=-1)

		# print(assigned_cluster, temp_center)
		self.centers[assigned_cluster] = temp_center
		# print(np.linalg.norm(self.centers, axis=-1).sum())

		return prediction == branch_is_taken

	def finish(self, num_predictions):
		print("====")
		print(self.labels)
		assert sum([e[False] + e[True] for e in self.labels]) == num_predictions
		print("====")

class S_Kmeans2(S_Kmeans):
	# https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm
	def __init__(self, **args):
		super().__init__(**args)
		self.alpha = 0.5
		# print("Clusters2, ", self.centers, self.alpha)

	def make_prediction(self, address, branch_is_taken):

		address = vectorize(address)
		prediction = self.default_pred

		distances = np.linalg.norm(address - self.centers, ord=2, axis=1.) 
		assigned_cluster = np.argmin(distances)

		stats = self.labels[assigned_cluster]
		
		prediction = stats[True] >= stats[False] 

		self.labels[assigned_cluster][branch_is_taken] += 1
		
		temp_center = self.centers[assigned_cluster]
		temp_center = temp_center + self.alpha * (address - temp_center)
		temp_center = temp_center / np.linalg.norm(temp_center, axis=-1)

		# print(assigned_cluster, temp_center)
		self.centers[assigned_cluster] = temp_center
		# print(np.linalg.norm(self.centers, axis=-1).sum())

		return prediction == branch_is_taken

class DenStream_Algo():
	# https://github.com/waylongo/denstream/blob/master/codes/DenStream.py
	def __init__(self, **args):
		# self.clusterer = DenStream(lambd=0.1, eps=0.5, beta=0.5, mu=3)
		self.clusterer = DenStream2(lambd=0.1, eps=0.5, beta=0.5, mu=3)
		
		self.default_pred = True

	def make_prediction(self, address, branch_is_taken):

		address = vectorize(address)

		predictions = self.clusterer.predict(address)
		self.clusterer.partial_fit(address, branch_is_taken)

		if predictions == None:
			predictions = self.default_pred

		return predictions == branch_is_taken
	
	def finish(self, num_predictions):
		print("====")
		print(f"Number of p_micro_clusters is {len(self.clusterer.p_micro_clusters)}")
		print(f"Number of o_micro_clusters is {len(self.clusterer.o_micro_clusters)}")
		print("====")	

class DenStream_Algo2():
	# https://github.com/waylongo/denstream/blob/master/codes/DenStream.py
	def __init__(self, **args):
		self.clusterer = DenStream3()
		
		self.default_pred = True

	def make_prediction(self, address, branch_is_taken):

		address = vectorize(address)

		predictions = self.clusterer.predict(address)
		self.clusterer.partial_fit(address, branch_is_taken)

		if predictions == None:
			predictions = self.default_pred

		return predictions == branch_is_taken
	
	def finish(self, num_predictions):
		print("====")
		print(f"Number of p_micro_clusters is {len(self.centers)}")
		print("====")	


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
        
def run_predictor(predictor, filename: str, return_detailed_output: bool = False) -> Tuple[int, int]:
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

    for i, (address, branch_is_taken) in enumerate(instructions):
        print(i / num_predictions, end="\r")
        prediction_is_correct = predictor.make_prediction(address, branch_is_taken)
        if not prediction_is_correct:
            num_mispredictions += 1

        if return_detailed_output and detailed_output:
            actual = branch_is_taken
            predicted = prediction_is_correct == branch_is_taken
            detailed_output[(actual, predicted)] += 1

    if return_detailed_output and detailed_output:
        return num_predictions, num_mispredictions, detailed_output

    return num_predictions, num_mispredictions


if __name__ == "__main__":
	cfg = get_args_parser()
	cfg = cfg.parse_args()
	function_name = cfg.algorithm_name
	trace_file = cfg.trace_file
	predictor = locals()[function_name](**cfg.additional_args)

	num_predictions, num_mispredictions = run_predictor(predictor, trace_file, )

	misprediction_rate = 100 * num_mispredictions / num_predictions

	print(f"number of predictions:		{num_predictions}")
	print(f"number of mispredictions:	{num_mispredictions}")
	print(f"misprediction rate:		{misprediction_rate:.2f}%")
	predictor.finish(num_predictions)
	
