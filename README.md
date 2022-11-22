
# Evaluating an HSIC Dependency Measure Based Novel Black-Box Attribution Method

We introduce new evaluation metrics, Average Stability and MuFidelity to evaluate
a novel attribution method, described in "Making Sense of Dependence: Efficient 
Black-box Explanations Using Dependence Measure". We also try different sampling 
techniques to understand the changes in the behaviour of the new attribution method.

To run the code, kindly follow these steps:
1. run the command "chmod u+x code_run.sh" in the same directory as the files.
2. To run the bash file, kindly execute this command: './code_run.sh'.
3. To save the output to a text file, append '> output.txt' for convenience.

If this execution gives a TensorFlow Stream error, then the following steps need
to be followed:

The bash file includes installation of two libraries, followed by the execution of two py files.
The first py file named 'main.py' may be run on a GPU for speed but the second file
titled, 'mu_fid.py' must be run on the CPU.

Sorry for the inconveniences caused.
For any other issues faced, kindly drop an email at:
praharsh19@iiserb.ac.in OR devesh19@iiserb.ac.in
