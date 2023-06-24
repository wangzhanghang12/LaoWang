### 视觉中的Transformer

![image-20230623154201962](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230623154201962.png)

#### 自注意力机制 

 ![image-20230623155214758](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230623155214758.png)

传统的NLP用的是RNN

### multi head attention

多头自注意力(Multi-Head Attention)是Transformer模型中的一项关键机制。它允许模型 jointly 关注输入序列的不同方面。具体来说,多头自注意力包含多个并行的注意力头(通常是8个或更多),每个注意力头都有自己的WQ,WK和WV矩阵。这些注意力头一起对输入序列进行处理,产生最终的注意力输出。多头注意力的计算过程如下:

1.为每个注意力头计算注意力,得到一系列注意力输出:

`AttentionHeadi = Attention(QWiQ, KWiK, VWiV)`

其中QWiQ,KWiK和VWiV是第i个注意力头的WQ,WK和WV矩阵。

2.将这些注意力头的输出拼接成一个tensor:

`MultiHead(Q, K, V) = Concat(AttentionHead1, AttentionHead2, ..., AttentionHeadh)`

3.加一层线性映射:`Output = WOMultiHead`

其中WO是一个可训练矩阵。这样,每一个注意力头都专注于输入的不同方面,它们一起产生最终的注意力表达,这可以比单一注意力更好地建模输入的不同语义。

多头自注意力的主要优点:

1.可以关注输入序列的不同方面(不同的注意力头可以学习不同的构造),产生更丰富的表达。

2.防止注意力机制过于集中于输入序列的某一部分(不同的注意力头会建立不同的注意力中心)。

3.通过使用多个注意力头并结合其输出,可以在一定程度上控制预测variance。

4.并行运算,计算速度更快。多头自注意力已被证明在Transformer和BERT等模型中是非常有效的机制。它是实现深度注意力和建模复杂输入间关系的有力工具。

![image-20230623160052390](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230623160052390.png)

RNN通常是时间域上的而CNN通常是在空间域上的

RNN是迭代的,但是CNN很好并行

![image-20230623160505918](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230623160505918.png)

###  方法一:Visual Attention Layer

可视注意力层是一种用于图像Caption生成的注意力机制。它允许模型在生成每个词元时关注输入图像的不同区域,产生更准确和详细的图像描述。典型的图像Caption模型包含:

1. 图像编码器:使用CNN编码输入图像,得到一系列特征图。
2. RNN解码器:使用LSTM/GRU从词元序列生成描述。
3. 注意力层:连接图像编码器和RNN解码器。当RNN解码器生成每个词元时,注意力层决定RNN应该关注哪些图像区域,并产生相应的上下文向量

可视注意力层的工作原理如下: 

1.RNN解码器的当前隐藏状态ht作为"查询"用于注意力计算。

2.图像编码器的特征图F作为"键"(K)和"值"(V)用于注意力计算。

3.计算注意力权重α,表示RNN应该关注图像的哪些区域:`α = softmax(htFKT)`其中FKT是K的转置。

4.产生上下文图像表达ct作为ht和α的加权平均:`ct = αFKV` 

5.ct会与ht一起作为输入馈入RNN解码器生成下一个词元。

6.这个过程在生成每个词元时重复,实现对输入图像区域的动态选择。

可视注意力层的主要优点:

1.使图像Caption生成过程更加数据驱动,模型可以决定在生成每个词元时最相关的图像区域。

2.产生更准确和详细的图像描述。模型不需要简单地关注整个图像,而可以选择最相关的区域。

3.提高模型的解释性。我们可以通过观察模型的注意力权重α来理解其关注的图像区域及生成相应词元的原因。可视注意力已被证明可以有效地提高图像Caption模型的性能。它实现了对图像的动态选择性编码,为RNN生成每个词元提供最相关的上下文信息。

### 方法二:Vision Transformer

Vision Transformer(ViT)是一种基于Transformer的图像分类模型。与CNN不同,它直接对图像的patch进行变换和编码,不依赖于卷积运算。

ViT的主要思想是:将图像分割成 fix size 的 patch,将每个patch视为一个"词元"输入到Transformer中,对其进行编码和分类预测。这使得ViT可以完全基于注意力机制来学习图像的空间关系和分类特征。

具体来说,ViT包含以下主要模块:

1.Patch分割:将输入图像分割为n x n的patch,然后平铺成序列输入到Transformer。

2.位置编码:为每个patch的序列中每个patch添加位置编码,表示其在2D平面中的位置。

3.Transformer编码器:使用Encoder包含N个TransformerBlock,每个TransformerBlock包括Multi-Head Attention和MLP。这 encoder patch序列并学习其关系。

4.分类器:使用 Encoder的输出以及一个多层感知器(MLP)进行图像分类预测。

5.残差连接和层归一化:像Transformer一样,在每个模块中使用。

ViT的优点在于:

1. 完全基于注意力机制,可以直接编码图像的2D结构关系,不依赖卷积。这种新视角可能会产生突破。
2. 比CNN更容易并行化和跨设备部署。注意力机制更适合在GPU和TPU上高效计算。
3. 可解释性更强。我们可以通过可视化注意力权重矩阵来理解模型关注的图像区域和其关系。
4.  可扩展性更强。增加Transformer的块数、注意力头数和编码器深度可以轻松构建更深入和强大的模型。

然而,ViT也面临以下挑战:

1. 较高的计算复杂度,尤其是当序列长度较长时(高分辨率图像)。
2. 存在" Institute of Electrical and Electronics Engineers(IEEE)"偏差,即中间位置的patch过于关注。位置编码可以部分缓解这一问题。
3. 此模型未能有效利用图像的局部结构(无卷积),效果可能不如CNN。ViT是一种全新的基于Transformer的图像分类方法,代表着基于注意力的视觉学习新思路。它可用于informer图像分类和语义分割等任务。但也面临一定的技术挑战,需要更多实验来评估其效果。

![image-20230623162451535](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230623162451535.png)

 

![image-20230623163458216](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230623163458216.png)