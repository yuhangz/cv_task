# cv_task

<img src="https://img.shields.io/badge/python-blue?&style=for-the-badge&logo=python&logoColor=white" style="margin-right: 0.25rem"/> <img src="https://img.shields.io/badge/jupyter notebook-orange?&style=for-the-badge&logo=python&logoColor=white" style="margin-right: 0.25rem"/>

## 介绍
这是一次CV作业的baseline，实现了对EMNIST数据集中的balanced数据集进行分类的任务。
修改config文件夹中的文件以更改模型，目前记录用tensorboard记录了loss-epoch，在SCALARS中查看，输出了测试集中47类的准确率以及测试集前20张图片的分类结果。


## 文件结构

```
📁code
   ├─📁config 运行实验的配置文件
   ├─📁data 数据
   ├─🐍CNN.ipynb 运行 CNN 网络的 notebook
   ├─🐍MLP.ipynb 运行 MLP 网络的 notebook
   ├─🐍param_search_cnn.py 搜索 CNN 最优超参数的程序
   ├─🐍param_search_mlp.py 搜索 MLP 最优超参数的程序
   └─🐍type.py
📁figure    用于论文组进行写作的插图
📄report.md 用于进行团队交流的文档
```

## 运行

```bash
git clone https://github.com/yuhangz/cv_task.git && cd cv_task
pip install -r requirements.txt
cd code
# 搜索 CNN 的超参
python param_search_cnn.py
# 搜索 MLP 的超参
python param_search_mlp.py
```