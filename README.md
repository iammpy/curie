# CURIE评测使用指南

## 1、环境安装

conda create -n curie python=3.10  
conda activate curie  
切换到项目根目录
pip install -r requirements.txt

## 2、启动评测

### 启动脚本位置 
curie/colabs/CURIEbenchmark_inference_Command_R_Plus.py

### 命令行参数传递：
python colabs/CURIEbenchmark_inference_Command_R_Plus.py 模型名 模型url 并发数(可选，默认32)  


### 输出结果 
mpve任务：
curie/inference/multi_runs/current/mpve/mat_paper_to_property_1_shot_exclude_trivia/模型名
dft任务：
curie/inference/multi_runs/current/dft/extract_dft_metadata_1_shot/模型名

后缀为ave_accuracy的json文件中存储整体的accuracy、recall、f1分数，这里面的f1分数是最终的指标

### 注意事项
模型的调用使用固定形式，如需要修改，可修改curie/colabs/model.py 文件中的call_server函数
