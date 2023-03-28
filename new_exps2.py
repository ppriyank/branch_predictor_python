import argparse
from mmcv import DictAction
from typing import List, Tuple
import os 
from sklearn.cluster import MiniBatchKMeans
import numpy as np 
import random 

# from clusopt_core.cluster import CluStream
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.size'] = 16
# plt.rcParams["legend.labelcolor"] = "black"
# plt.rcParams["legend.edgecolor"] = "black"
# plt.fig.subplots_adjust() 
plt.figure(figsize=(10, 10))

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

class SOS_Cluster():
	# https://github.com/ruteee/SOStream/blob/master/notebooks/SOStream%20Teste.ipynb
	def __init__(self, past=None, dim=22, k=5):
		from SOStream.sostream import SOStream
		self.sostream_clustering = SOStream(alpha = 0, min_pts = 3, merge_threshold = 50000)
		self.default_pred = True

	def make_prediction(self, address, branch_is_taken):
		
		address = vectorize(address)

		prediction=  self.sostream_clustering.predict(address)
		if prediction is None:
			prediction = True
			

		label = (branch_is_taken * 1.0 - 0.5 ) * 2
		self.sostream_clustering.process(address, label)
		
		return prediction == branch_is_taken
		
	
	def finish(self, num_predictions):
		print("====")
		print(f"Buffer:  {len(self.sostream_clustering.M)}")
		print("====")	

class clusopt():
	# https://pypi.org/project/clusopt-core/
	def __init__(self, **args):
		import pdb
		pdb.set_trace()
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

class Plot():
	# https://pypi.org/project/clusopt-core/
	def __init__(self, **args):
		from sklearn.manifold import TSNE
		
		self.tsne = TSNE(n_components=2, random_state=0)
		self.data = []
		self.labels = [] 
	
	def make_prediction(self, address, branch_is_taken):

		address = vectorize(address)
		self.data.append(address)
		self.labels.append(branch_is_taken)

		return False
	
	def finish(self, num_predictions, file_name="chikka.pkl"):
		self.data = np.concatenate(self.data, axis=0)
		self.labels = np.array(self.labels)
		import pdb
		pdb.set_trace()
		X = self.tsne.fit_transform(self.data)
		saved_obj = dict(X = X, Y=self.labels)
		with open(file_name, 'wb') as f:
			pickle.dump(saved_obj, f)    
		# plt.clf()  
  #   	plt.grid(alpha=0.5)
  #   	color_class1 = {0:"black" , 1:"red"}
	 #    for i,cl in enumerate({0, 1}):
	 #        if cl == 'controlled' :
	 #            color = color_class1
	 #        else:
	 #            color = color_class2
	 #        indices = conditions == cl
	 #        vals = X[indices]
	 #        subs = subjects[indices]
	 #        print(indices.sum(), vals)
	 #        # scatter_plot = plt.scatter(vals[:,0], vals[:,1], color=colors[i], alpha=alphas[s], label=cl)
	 #        for k,s in enumerate(sub_ids):
	 #            subset_index = subs == s
	 #            # vals = vals / np.linalg.norm(vals, axis=1).reshape(-1, 1)
	 #            scatter_plot = plt.scatter(vals[subset_index,0], vals[subset_index,1], color=color[k], alpha=alpha)
	 #            # plt.colorbar(label="Like/Dislike Ratio")
	 #    # plt.legend()
	 #    plt.title(f"t-SNE Plot : Controlled vs Treatment")
	 #    plt.savefig("fig1.png")
	 #    plt.clf()  

		# import pdb
		# pdb.set_trace()
		
class Plot2(Plot):
	# https://pypi.org/project/clusopt-core/
	def __init__(self, **args):
		from sklearn.decomposition import PCA
		self.pca = PCA(n_components=2)
		self.data = []
		self.labels = [] 
		
	def finish(self, num_predictions, file_name="PCA.pkl"):
		if os.path.exists(file_name):
			with open(file_name, 'rb') as f:
				saved_obj = pickle.load(f)    
			X = saved_obj["X"]
			Y = saved_obj["Y"]
		else:
			self.data = np.concatenate(self.data, axis=0)
			self.labels = np.array(self.labels)
			X = self.pca.fit_transform(self.data)
			Y = self.labels
			saved_obj = dict(X = X, Y=self.labels)
			with open(file_name, 'wb') as f:
				pickle.dump(saved_obj, f)    
		
		plt.clf()
		plt.grid(alpha=0.5)
		color_class = {0:"black" , 1:"red"}
		Y= Y * 1.0
		for i,cl in enumerate({0, 1}):
			color = color_class[cl]
			indices = Y == cl
			vals = X[indices]
			lab = Y[indices]
			scatter_plot = plt.scatter(vals[:,0], vals[:,1], color=color, alpha=0.1, label=cl)
		plt.legend()
		plt.title(f"PCA Plot")
		plt.savefig("fig1.png")
		plt.clf()  

class Nearest_Neighbour():
	def __init__(self, past=None, dim=22, k=5):
		self.default_pred = True
		self.centers = [float(self.default_pred) for i in range(past)] 
		self.distances = np.array([[0 for i in range(dim)] for j in range(past)])
		self.past = past
		self.k = k 

	def make_prediction(self, address, branch_is_taken):
		
		prediction = self.default_pred
		address = vectorize(address)
		
		dist = ((address - self.distances) ** 2).sum(-1)
		ordered = np.argsort(dist)
		# sorted_neigh = sorted(dist, key=lambda x: x[1])[:n_neighbors]

		self.distances = self.distances[1:]
		self.centers.pop(0)

		self.distances = np.append(self.distances, address, axis=0)
		self.centers.append(branch_is_taken * 1.0)
		
		
		return prediction == branch_is_taken
	
	def finish(self, num_predictions):
		print("====")
		print(f"Buffer:  {self.centers}")
		print("====")	

class River_log():
	# pip install pep517 jsonpatch
	# pip install git+https://github.com/online-ml/river --upgrade
	# https://github.com/online-ml/river
	
	def __init__(self, category="None", dim=22):
		from river import compose, linear_model, metrics, preprocessing, forest
		# river/river/forest/
		self.dim = dim
		self.default_pred = True

		self.vectorize_fn = self.vectorize2
		self.metric = metrics.Accuracy()
		# metric = metrics.MacroF1()
		if category == "logistic":
			self.model = compose.Pipeline(linear_model.LogisticRegression())
		elif category == "Perceptron":
			self.model = compose.Pipeline(linear_model.Perceptron())
		elif category == "ALMA":
			self.model = compose.Pipeline(linear_model.ALMAClassifier())
		elif category == "ARFClassifier":
			self.vectorize_fn = self.vectorize
			self.model = compose.Pipeline(
				# preprocessing.StandardScaler(),
				forest.ARFClassifier()
			)
		elif category == "AMFClassifier":
			self.vectorize_fn = self.vectorize
			self.model = compose.Pipeline(
				# preprocessing.StandardScaler(),
				forest.AMFClassifier()
			)		
		# preprocessing.StandardScaler(),
	def vectorize(self, address, no_normalize=True):
		addr = bin(address)[2:]
		addr = {i: int(addr[i]) for i in range(self.dim)}
		return addr

	def vectorize2(self, address, no_normalize=True):
		addr = bin(address)[2:]
		addr = [int(e) for e in addr]
		denom = sum(addr) ** 0.5
		addr = {i: addr[i] / denom for i in range(self.dim)}
		return addr

	def make_prediction(self, address, branch_is_taken):
		
		address = self.vectorize_fn(address)
		prediction = self.model.predict_proba_one(address)
		pred = prediction[True] > prediction[False]
		
		self.model.learn_one(address, branch_is_taken)

		return pred == branch_is_taken
	
	def finish(self, num_predictions):
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
	
