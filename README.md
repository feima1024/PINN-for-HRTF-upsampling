This is the source code for paper 

"Spatial Upsampling of Head-Related Transfer Functions Using a Physics-Informed Neural Network"
which is also attached here as a PDF file. 



Sec. V, the interpolation experiment 

Read pinn.py and see the result shown in  'interpolation.png' and 'interpolation.fig' 

-------------------------------------------------------------------------------------------------
Or you can run the code and generate the result by yourself: 

0, Download all files into one folder; 

1, start a python terminal;

2, go to the same folder;   

3, run the code by exec(open('pinn.py').read()); 

4, pinn.py will read the 40.mat file and generate the 40_L3.mat.  


Trained on one cpu core of Macbook M1 pro, the runtime was about 7 hours.  
The runtime can be significantly reduced with more cpu cores.  


5, Start Matlab, and go to the same folder, run the 'result.m' file, then you will get the figures. 

-------------------------------------------------------------------------------------------------


