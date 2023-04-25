make

sim smith 2 traces/gcc_trace.txt
sim bimodal 5 traces/gcc_trace.txt

sim gshare 8 6 traces/gcc_trace.txt

sim hybrid 5 4 4 4 traces/gcc_trace.txt


sim smith 2 traces/gcc_trace.txt > chikka.txt
sim bimodal 5 traces/gcc_trace.txt > chikka.txt
sim gshare 8 6 traces/gcc_trace.txt > chikka.txt
sim hybrid 5 4 4 4 traces/gcc_trace.txt > chikka.txt