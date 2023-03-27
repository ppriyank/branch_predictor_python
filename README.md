# branch_predictor_python
Branch Prediction : Python 


## Smith n-bit predictor

```
python smith.py 3 traces/gcc_trace.txt
python smith.py 1 traces/jpeg_trace.txt
python smith.py 4 traces/perl_trace.txt
```

## Bimodal n-bit predictor

```
python bimodal.py 6 traces/gcc_trace.txt
python bimodal.py 12 traces/gcc_trace.txt
python bimodal.py 4 traces/jpeg_trace.txt
```

## Gshare (m,n) predictor

```
python gshare.py 9 3 traces/gcc_trace.txt
python gshare.py 14 8 traces/gcc_trace.txt
python gshare.py 11 5 traces/jpeg_trace.txt
```

### BenchMarking

```
| File | Algo | N |Acc | 
| --- | --- | --- | 
| traces/gcc_trace.txt | smith.py | 1 | 45.24 |
| traces/gcc_trace.txt | smith.py | 2 | 43.01 | 
| traces/gcc_trace.txt | smith.py | 3 | 41.57 | 
| traces/gcc_trace.txt | smith.py | 4 | 41.68 |
| traces/gcc_trace.txt | bimodal.py | 2 | 38.82 |
| traces/gcc_trace.txt | bimodal.py | 4 | 35.05 | 
| traces/gcc_trace.txt | bimodal.py | 6 | 30.29 | 
| traces/gcc_trace.txt | bimodal.py | 8 | 21.66 | 
| traces/gcc_trace.txt | bimodal.py | 10 | 15.33 | 
| traces/gcc_trace.txt | bimodal.py | 12 | 12.30 |
| traces/gcc_trace.txt | gshare.py | 2,2 | 40.78% | 
| traces/gcc_trace.txt | gshare.py | 3,3 | 39.56% | 
| traces/gcc_trace.txt | gshare.py | 4,4 | 38.34 | 
| traces/gcc_trace.txt | gshare.py | 9,3 | 20.88 | 
| traces/gcc_trace.txt | gshare.py | 14,8 | 11.62 | 
| traces/gcc_trace.txt | gshare.py | 14,8 | 16.43 | 
```


### Streaming Algorithms
    - https://arxiv.org/pdf/2007.10781.pdf

`chameleon_cluster`: Doesnt seem be to totally online     

    - https://github.com/Moonpuck/chameleon_cluster/blob/1c0a65ee6a79706e4d415dd7ca78da5d3c29906d/chameleon.py#L80

### ToDO    
    - https://pypi.org/project/clusopt-core/
    
    - https://github.com/ruteee/SOStream/blob/master/notebooks/SOStream%20Teste.ipynb

    - https://github.com/online-ml/river

    - Wrong Pendalty = -2
    - Correct Reward = 1
