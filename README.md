# BERT-CRF-CRE 金融领域因果关系抽取模型

## 1.环境安装 
- Python版本支持：3.5 或者 3.6 或者 3.7
```
python -m pip install -r requirements.txt
```

## 2.模型参数
从百度网盘将模型参数文件下载，并存放置项目bert_cause/config文件夹下

- 链接：https://pan.baidu.com/s/16ms_v_l11TO6zRrPkBYV1Q 提取码：bert 

- 模型参数包含三个文件：
```
model.ckpt.data-00000-of-00001
model.ckpt.index
model.ckpt.meta
```
- 运行时需指定参数文件名
```
-init_checkpoint=model.ckpt
```

## 3.模型运行
①开启tcp服务
```
python server_run.py
```
- 支持参数，若不指定则采用等号右边的默认值
```
-root_path=.                    # 模型工作路径，也可设置成绝对路径如/root/BERT-CRF-CRE
-init_checkpoint=model.ckpt     # 模型加载的参数检查点
-port=5555                      # tcp服务接收端口
-port_out=5556                  # tcp服务发送端口
-timeout=-1                     # 最大响应时间
-host=localhost                 # tcp服务主机名
```
- 注意：需要等到控制台打印出Read and listening之后再开启预测

②开启预测
```
python predictor.py
```
- 支持参数，若不指定则采用等号右边的默认值
```
-root_path=.                        # 模型工作路径，也可设置成绝对路径如/root/BERT-CRF-CRE
-port=5555                          # tcp服务接收端口
-port_out=5556                      # tcp服务发送端口
-timeout=-1                         # 最大响应时间
-host=localhost                     # tcp服务主机名
-use_center=True                    # 预测结果是否按照中心词匹配
-inpur_file=pred_input.txt          # 输入的预测文件名
-result_to_file=False               # 是否将结果输出到文件
-output_file=pred_output.txt        # 输出的结果文件名
```

# 附录
如果需要进行因果分类的以及开启http服务的，可以按照下面的步骤进行。

## 1.模型参数
下载分类模型参数，从百度网盘将模型参数文件下载，并存放置项目bert_theme/config文件夹下。

- 链接：https://pan.baidu.com/s/1ImCW_j4Uj-cgbx0tFbgutQ 提取码：them 

- 分类模型参数包含三个文件：
```
model.ckpt-4788.data-00000-of-00001
model.ckpt-4788.index
model.ckpt-4788.meta
```

- 运行时需指定参数文件名
```
-init_checkpoint=model.ckpt-4788
```

## 2.模型运行
①启动因果抽取模型tcp服务
```
python server_run.py
```
- 支持参数，若不指定则采用等号右边的默认值
```
-root_path=.                    # 模型工作路径，也可设置成绝对路径如/root/BERT-CRF-CRE
-init_checkpoint=model.ckpt     # 模型加载的参数检查点
-port=5555                      # tcp服务接收端口
-port_out=5556                  # tcp服务发送端口
-timeout=-1                     # 最大响应时间
-host=localhost                 # tcp服务主机名
```

②启动因果分类模型tcp服务
```
python server_theme_run.py
```
- 支持参数，若不指定则采用等号右边的默认值
```
-root_path=.                        # 模型工作路径，也可设置成绝对路径如/root/BERT-CRF-CRE
-init_checkpoint=model.ckpt-4788    # 模型加载的参数检查点
-port=5557                          # tcp服务接收端口
-port_out=5558                      # tcp服务发送端口
-timeout=-1                         # 最大响应时间
-host=localhost                     # tcp服务主机名
```

## 3.开启http服务
http服务需要等待两个tcp服务开启后控制台输出ready and listening之后再开启。
```
python http_run.py
```
- 支持参数，若不指定则采用等号右边的默认值
```
-host=localhost         # http服务主机名
-http_port=8555         # http服务端口
-theme_port=5557        # 分类模型tcp服务接收端口
-theme_port_out=5558    # 分类模型tcp服务发送端口
-cause_port=5555        # 抽取模型tcp服务接收端口
-cause_port_out=5556    # 抽取模型tcp服务发送端口
-timeout=-1             # 最大响应时间
```
