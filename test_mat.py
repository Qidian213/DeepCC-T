import matlab.engine

from TestCode import mat 

eng = matlab.engine.start_matlab()
eng.addpath(r'TestCode/',nargout=0)

mat.printmat(eng)
