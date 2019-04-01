import matlab.engine
eng = matlab.engine.start_matlab()
ret = eng.py_mat(1.0,5.0)
print(ret) # 2.5
