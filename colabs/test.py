import os
if __file__ :
    # 表示当前是一个py文件
    os.chdir(os.path.dirname(__file__))
    print("当前路径:", os.getcwd())
    print(os.path.abspath('./'))