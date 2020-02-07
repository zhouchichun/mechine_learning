# mechine_learning
关于机器学习的整理
# 有监督学习算法

- 分类任务的实质。从原始特征中找到能够解释标签的有效特征。
因此，特征提取还有提取特征到标签的映射是两个重要环节。
特征提取如统计提取方法，只能通过一定范围内遍历，寻找有效的特征。
然而，传统的统计特征提取算法遇到了巨大的难题，算力不够，速度慢。

深度学习集特征提取和映射两个重要环节于一体。

## 机器学习
- *熵*。度量分布所携带的信息程度。
在完全没有信息时，分布是均匀的，此时熵最大。如，在没有先验信息的情况下，需要在大草原上打猎，
那么均匀分配人力把守不同地方是最好的策略。
但是如果知道猎物只在少数几个地方出现，那么我们就
会集中在这个地方狩猎，粗略的说，前者熵大，信息少，后者熵小，信息大。
<center>
<img src="D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\shang.png"></img>
</center>

- *Gini指数*。衡量数据的不纯度或者不确定性。如果数据是纯的，那么Gini数是1。
在分类问题中，假设有K个类，样本点属于第k类的概率为Pk，
则概率分布的gini指数的定义为：
<center>
<img src="D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\gini.png"></img>
</center>

- *信息增益*。它度量了在得知 X 以后使得 Y 不确定性减少程度，
也就是 X 为我们预测 Y 提供了多少有用的信息。
这个度量叫做信息增益，也称为互信息。

- *集成*：对其他算法进行组合的一种形式。
  
  - 投票算法（bagging）：基于数据随机重抽样分类器构造的方法。例如随机森林（random forest）
  
  - 再学习（boosting）:基于所有分类器的加权求和方法。
  
  - bagging和boosting的区别：1）bagging是一种与boosting所使用的多个分类器的类型
  （数据量和特征量）都是一致的。2）bagging是由不同的分类器（数据和特征随机化）经过训练，
  综合得出的出现最多的分类结果；boosting是通过调整已有分类器错分的
  那些数据来获得新的分类器，得出目前最优的结果。3）bagging中的分类器权重是
  相等的；而boosting中的分类器加权求和，所以权重并不相等，
  每个权重代表的是其对应分类器的上一轮迭代中的成功过度。
- *判别模型与生成模型*
  - 判别模型，对条件概率建模，根据x预测y。
  
  - 生成模型，给出x和y的联合概率密度。
 <center>
<img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\panbie.jpg'></img>
</center>

- *PCA*：通过特征的线性变换，将协方差矩阵对角化，保留协方差矩阵特征值最大的特征向量，
以此为新的特征。

  - 在原特征下计算协方差矩阵。
  - 对角化协方差矩阵。
  - 找到大特征值对应的特征向量。

```
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)

n_components: int, float, None 或 string，PCA算法中所要保留的主成分个数，也即保留下来的特征个数，如果 n_components = 1，将把原始数据降到一维；如果赋值为string，如n_components='mle'，将自动选取特征个数，使得满足所要求的方差百分比；如果没有赋值，默认为None，特征个数不会改变（特征数据本身会改变）。
copy：True 或False，默认为True，即是否需要将原始训练数据复制。
whiten：True 或False，默认为False，即是否白化，使得每个特征具有相同的方差。

```
[参考](https://scikit-learn.org/)


- *条件随机场、隐马尔科夫等* [参考](https://zhuanlan.zhihu.com/p/28305337)
  
  - 概率图模型是一类用图来表达变量相关关系的概率模型。
  概率图模型可大致分为两类：使用有向无环图表示变量间的依赖关系，
  称为有向图模型或贝叶斯网；使用无向图表示变量间的相关关系，
  称为无向图模型或马尔可夫网。按照概率图模型将HMM，MEMM，
  CRF放在这里一起对比，都是通过隐藏状态，
  历史可观测状态预测未来可观测状态。
  
  - *HMM* 可观测序列O，隐藏序列H，隐藏序列之间的转移密度矩阵，从隐藏序列到可观测序列的转换序列
  <center>
  <img src="D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\hmm.png">
  </img>
  </center>
 
  - *MEMM* MEMM是这样的一个概率模型，即在给定的观察状态和前
  一状态的条件下，出现当前状态的概率。
<center>
  <img src="D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\emm.png">
  </img>
  </center>
  
  - *CRF* 是无向图
  <center>
  <img src="D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\crf.png">
  </img>
  </center>
  - 总结：HMM模型中存在两个假设：一是输出观察值之间严格独立，
  二是状态的转移过程中当前状态只与前一状态有关。但实际上序
  列标注问题不仅和单个词相关，而且和观察序列的长度，
  单词的上下文，等等相关。MEMM解决了HMM输出独立性假设的问题。
  因为HMM只限定在了观测与状态之间的依赖，而MEMM引入自定
  义特征函数，不仅可以表达观测之间的依赖，还可表示当前观
  测与前后多个状态之间的复杂依赖。CRF不仅解决了HMM输出独立性假设的问题，
  还解决了MEMM的标注偏置问题，MEMM容易陷入局部最优是因为只在局部
  做归一化，而CRF统计了全局概率，在做归一化时考虑了数据在全局的分布，
  而不是仅仅在局部归一化，这样就解决了MEMM中的标记偏置的问题。
  使得序列标注的解码变得最优解。
  
<center>
<img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\hmmemm.png' ></img>
</center>

- *决策树*：决策树是一种靠构建树形结构和阈值进行分类的算法。
其中每个内部节点表示一个属性上的判断，
每个分支代表一个判断结果的输出，最后每个叶节点代表一种分类结果。因此决策树的关键是
生成树，确定节点的阈值。分裂的节点就是样本的某个维度的属性。把特征空间划分为一系列的矩形区域。 
如，一个二叉树，包含A,B,C,D阈值。
<center>
<img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\jueceshu.jpg' style='width: 400;height:300'></img>
</center>
 - 生成树：当一个节点所代表的属性无法给出判断时，则选择将这一节点分成2个
子节点（如不是二叉树的情况会分成n个子节点）。
*1）ID3*: 对某个属性，按照某个阈值分类，
分错最少，则该属性做父节点。ID3 算法实质是用信息增益来判断当前节点应
该用什么特征来构建决策树。在相同条件下，取值比较多的特征比取值少的特征信息增益大。
存在过拟合问题；存在特征取值多的信息增益大。
*2）C4.5*：优化项要除以分割太细的代价，这个比值叫做信息增益率。
*3）CART*只能将一个父节点分为2个子节点。CART用GINI指数来决定如何分裂。
总体内包含的类别越杂乱，GINI指数就越大（跟熵的概念很相似）

 - 选择适当的阈值使得分类错误率最小。可以以gini参数，mse，熵等作为优化的指标
 进行参数和数结构的优化。
```
DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                       random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                       min_impurity_split=None, class_weight=None, presort=False)
DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, 
                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                      random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                      min_impurity_split=None, presort=False)
```
[参考](https://scikit-learn.org/)

<center>
<img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\sklearn_jueceshu.jpg' style='width: 750;height:788'></img>
</center>

- *随机森林*：随机森林的结果是依赖于多棵决策树的结果，这是一种集成学习的思想。
对于每棵树都有放回的随机抽取训练样本，这里抽取随机抽取部分的样本作为训练集，
再有放回的随机选取m个特征作为这棵树的分枝的依据，这里要注意。
这就是“随机”两层含义，一个是随机选取样本，一个是随机选取特征。通过
检验有和没有某个特征的树给出的袋内袋外误差来判断某个特征的重要程度。

```
forest = RandomForestRegressor(n_estimators=’warn’,criterion=’gini’, max_depth=None,
                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
							   max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
							   min_impurity_split=None, bootstrap=True, oob_score=False, 
							   n_jobs=None, random_state=None, verbose=0, warm_start=False, 
							   class_weight=None)

重要参数:
		1、n_estimators：森林里的数目数量；默认=10
		2、 max_depth：树的最大深度
		3、min_samples_split：拆分内部节点所需要的最小样本数；默认=2
		4、min_samples_leaf：叶子节点所需要的最小样本数
		5、oob_score：是否使用袋外样本来估计泛化精度
		6、n_jobs：适合和预测并运行的作业数；默认=None；-1表示使用所有的处理器
重要方法：
		1、fit（X，y）：从训练集（X，y）建立一片森林
			X：训练集样本数据
			y：目标值（分类中的类标签，回归中的实数）
		2、predict（X ）：预测X的类
			X：输入的样本数据
			return：预测的类
		3、predict_proba（X ）
			X：输入的样本数据
			return：预测为每个类的概率	

```
[参考](https://scikit-learn.org/)

- SVM，在参数空间中，找到一个曲面，满足这个曲面可以将样本分开。
问题是一个有约束的凸优化问题，[详见](https://zhuanlan.zhihu.com/p/31886934)
```
class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,
                      probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
					  max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
```
[参考](https://scikit-learn.org/)


## 神经网络

- *MLP*。万能近似定理证明了前馈神经网络（多层感知机或者全连接）
在满足一定结构的情况下，能够以任何精度拟合任何函数。MLP的强大功能就是可以
拟合出从特征到标签的函数映射关系。

```
 x = tf.placeholder(tf.float64, [None,10])
 w = tf.get_variable(name='w', 
                    shape=[10, 128], 
                    initializer=tf.contrib.layers.xavier_initializer(), 
                    dtype=tf.float64)
 b = tf.get_variable(name='b', 
                    shape=[128], 
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float64)

 hidden=tf.sigmoid(tf.matmul(x,w)+b)

 y=tf.matmul(hidden,w2)+b2

```

- 特征提取模块。特征提取模块由于其参数是科学的，因此具有强大的特征提取功能，不同的方式
对于不同类型的特征有效。如，环境有关特征，环境无关特征等等。

  - *CNN* 通过卷积核提取原始特征中包含的与空间位置无关的有效特征。
  
  ```
  input=tf.placeholder(tf.float64, [None,1024,1024,3])
  out=tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    ...
    name=None)
  
  ```
  [参考](https://tensorflow.google.cn/api_docs/)
  
  - *RNN* 通过引入一个共用参数，提取序列中与顺序有关的有效特征。
  是一种编码方法，因此可以用于翻译等问题。
  输入具有序列的概念。每个时间点上的特征经过一个MLP得到输出的特征。
  引入一个共用的隐藏参数，隐藏状态也通过MLP得到输出特征。将输入特征与共用隐藏状态
  整合起来：下一个时刻特征与上一个时刻隐藏状态经过MLP的输入一起，得到下一时刻
  的输出。从而将序列中历史信息保存。见图
  <center>
  <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\rnn2.jpg'></img>
  </center>
  
  <center>
  <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\rnn1.jpg'></img>
  </center>
  
  - *门控RNN*:GRU、LSTM。在RNN中输出特征和记录历史信息的状态h是相同的。
  在RNN的基础上，将输出和记录历史信息的状态分开。引入不同的门，控制历史信息状态的值，如，
  是否遗忘，是否记忆等。
  以使不同历史长度的特征得到提取。
  <center>
  <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\gru0.png'></img>
   </center>
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\gru1.png'></img>
   </center>
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\gru2.png'></img>
   </center>
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\gru3.png'></img>
 </center>
   <center>
 <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\gru4.png'></img>
  </center>
  LSTM，更加复杂。除了有记录历史信息的状态h，还有一个细胞状态C。有遗忘门，记忆门控制细胞状态
  的更新，输出门控制h的更新方式。如下
  <center>
  <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\lstm0.jpg'></img>
   </center>
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\lstm1.jpg' style="width:660"></img>
   </center>
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\lstm2.jpg'></img>
   </center>
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\lstm3.jpg'></img>
 </center>
  
  - *Attention和TRANSFORMER* Attention机制通过权重，把注意力集中放在重要的点上，
  使重要的特征权重增加。
  <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\attention.png' style="width:660"></img>
 </center>
  
  通过注意力机制（’两两内积‘）对序列输入中，任意两两的关联的特征进行提取。
  self attention其每个特征的输出都是依靠该特征与其余特征的内积权重。
  也是一个编码方法，由于没有顺序问题，其可以做到其每个特征的输出都是依靠该特征与
  其余特征的内积权重。
  
  transformer中，将每一个输入通过三个线性映射变为query,value,key向量。
  一个特征的query向量和所有特征的key向量做内积，得到权重，然后用value向量加权，得到输出。
  是一个编码方法，由于没有顺序问题，其可以做到并行运算。效果和效率好于RNN。
  对每个输入得到query,value以及key
  <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\trans0.jpg'></img>
 </center>
 query和key做self attention
  <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\trans1.jpg'></img>
 </center>
 将value做权重求和
 <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\trans2.jpg'></img>
 </center>
  - *POOLING* 下采样的方式直接提取特征。对于某些与环境无关的特征，其具有较好的提取效果。
- *损失*
 - softmax + 交叉熵。分类任务，单分类
 - sigmoid + mse。binary_crossentropy。分类任务，多分类。
 - TCT。翻译任务，解决对齐问题。

- *优化算法*
 在优化算法中需要考虑从样本中得到的更新方向是否存在随机误差，
 针对不同的参数是否需要不同的更新步长，随着训练步数的增加，
 训练步长的调整问题等。能够自动调整相关参数的为自适应算法。
 如果需要更快的收敛，或者是训练更深更复杂的神经网络，需要用一种自适应的算法。
 
 - SGD(Stochastic gradient descent)与 BGD对所有样本计算梯度
 更新参数不同，SGD体现了随机选择样本，在这个样本中计算梯度，更新参数。
 有严重震荡现象。
 - Momentum。 引入momentum，表示要在多大程度上保留原来的更新方向，
 这个值在0-1之间
 
 - Nesterov Momentum
 
 - Adagrad 不同参数学习率问题，自适应地为各个参数分配不同学习率的算法
 - Adadelta 学习率与训练步数问题，解决其学习率是单调递减的，训练后期学习率非常小
 - RMSprop 
 - Adam 这个算法是另一种计算每个参数的自适应学习率的方法。
 除了像 Adadelta 和 RMSprop 一样存储了过去梯度的平方 
 vt 的指数衰减平均值 ，也像 momentum 一样保持了过去梯度 
 mt 的指数衰减平均值

-参数初始化
 参数一般赋予随机值，以初始化，在经验范围内，一般采用高斯、均匀等分布作为
 参数初始化方式，但是分布的参数需要指定，通过当前环境给出分布参数的方法比较好，比如
 
 ```
 tf.constant_initializer() 常数初始化
 tf.ones_initializer() 全1初始化
 tf.zeros_initializer() 全0初始化
 tf.random_uniform_initializer() 均匀分布初始化
 tf.random_normal_initializer() 正态分布初始化
 tf.truncated_normal_initializer() 截断正态分布初始化
 tf.uniform_unit_scaling_initializer() 这种方法输入方差是常数
 tf.variance_scaling_initializer() 自适应初始化
 tf.orthogonal_initializer() 生成正交矩阵
 Xavier均匀初始化
 ```

- 效果提升机制
 
  - 残差 skip connect,通过将经过不同处理环节得到的输出整合作为下一步的输入
  以充分学习有效特征，避免出现在深度网络中的学习偏差。
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\cancha.jpg'></img>
 </center>
 
    <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\cancha1.jpg' style="width:660"></img>
 </center>
  - High way 引入transfer gate 和一个carry gate，解决网络太深造成的有效特征丢失等问题。
  gate实际上就是权重，控制不同路径输出特征的重要性。即，让网络自己去学习是不是有信息丢失。
   <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\high.png' style="width:660"></img>
 </center>
  - Batch Normolization。通过对样本特征进行归一化处理，
  加快训练速度，这样我们就可以使用较大的学习率来训练网络，其效果是经过验证得到的。
  <center>
   <img src='D:\dalidaxue_scholar\多体系统+神经网络\机器学习算法总结\pic\batch.png' style="width:660"></img>
 </center>
  - Drop out。通过设置概率控制每个神经元是否输出，相当于得到了很多的
  子网络，进而有集成学习的效果，如避免过拟合，提高泛化能力等。
  - L1 和 L2 正则化。 是对参数的先验，认为最有参数值分布在0附近，或者是稀疏的。
  - 参数共享、下采样等。提升训练效率，同时不降低网络性能。
  
#无监督学习

##机器学习

##神经网络

- 一类自编码器


#强化学习
