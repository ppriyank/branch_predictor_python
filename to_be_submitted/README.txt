Hi! We wrote the project in python (which the professor approved),
so the makefile situation is a little weird. We think it should work as
specified in the project description, but if it doesnt the files can be 
called directly. The process is quite similar to the project specification.

To run a smith validation run, use:
python smith.py <B> <tracefile>

To run a bimodal validation run, use:
python bimodal.py <M2> <tracefile>

To run a gshare validation run, use:
python gshare.py <M1> <N> <tracefile>

And to run a hybrid validation run, use:
python hybrid.py <K> <M1> <N> <M2> <tracefile>


It's exactly the same as the specification, except 'sim' is replaced
with 'python' and all the predictors have '.py' appended to the end;
everything else is exactly the same! You can even redirect the output
and diff the result with the answer key the same way too.

The <tracefile> must be a valid absolute or relative file path, or it
must be the name of a file that exists in a relative 'traces' directory

This zip archive only contains the code necessary to run the validation runs.
For our full code, visit https://github.com/ppriyank/branch_predictor_python

Thanks!