


python new_exps.py "running_mean" 1 traces/gcc_trace.txt
python new_exps.py "running_mean" 2 traces/gcc_trace.txt
python new_exps.py "running_mean" 3 traces/gcc_trace.txt



python new_exps.py "running_mean2" traces/gcc_trace.txt 1 0.75 
python new_exps.py "running_mean2" traces/gcc_trace.txt 2 0.75
python new_exps.py "running_mean2" traces/gcc_trace.txt 3 0.75


python new_exps.py "Ensemble" traces/gcc_trace.txt 3 2


python new_exps.py "RS" traces/gcc_trace.txt 3

python new_exps.py "Random_Bimodal" traces/gcc_trace.txt 6 
