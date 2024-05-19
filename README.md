# cv_task
这是一次CV作业的baseline，实现了对EMNIST数据集中的balanced数据集进行分类的任务。
修改config文件夹中的文件以更改模型，目前记录用tensorboard记录了loss-epoch，在SCALARS中查看，输出了测试集中47类的准确率以及测试集前20张图片的分类结果。
```
tensorboard --logdir={your_logdir}
```
