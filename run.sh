


python new_exps.py "running_mean" 1 traces/gcc_trace.txt
python new_exps.py "running_mean" 2 traces/gcc_trace.txt
python new_exps.py "running_mean" 3 traces/gcc_trace.txt



python new_exps.py "running_mean2" traces/gcc_trace.txt 1 0.75 
python new_exps.py "running_mean2" traces/gcc_trace.txt 2 0.75
python new_exps.py "running_mean2" traces/gcc_trace.txt 3 0.75


python new_exps.py "Ensemble" traces/gcc_trace.txt 3 2


python new_exps.py "RS" traces/gcc_trace.txt 3

python new_exps.py "Random_Bimodal" traces/gcc_trace.txt 6 

python new_exps2.py --algorithm_name="S_Kmeans" \
--trace_file="traces/gcc_trace.txt" --additional_args n_cluster=2
2: 48.50
3: 47.96
4: 46.85
5: 47.02
6: 46.32

python new_exps2.py --algorithm_name="S_Kmeans2" \
--trace_file="traces/gcc_trace.txt" --additional_args n_cluster=2
2: 49.63
3: 49.51
4: 48.59
5: 49.37
6: 48.94

python new_exps2.py --algorithm_name="DenStream_Algo" \
--trace_file="traces/gcc_trace.txt" 
