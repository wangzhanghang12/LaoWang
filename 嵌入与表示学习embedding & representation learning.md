# 嵌入与表示学习embedding & representation learning

 **嵌入是一个数学名词可以用代数中的映射来理解它**(单射 双射 满射)

1. 单射（Injective）：也称为"一一对应"或"一对一映射"。一个函数 f: A → B 是单射的，如果对于任意不同的元素 a1 和 a2 属于集合 A，如果 f(a1) = f(a2)，则 a1 = a2。简而言之，每个 B 中的元素最多只能由一个 A 中的元素映射得到。可以理解为函数的输入不会发生冲突，不会出现不同的元素映射到同一个元素的情况。
2. 满射（Surjective）：也称为"到上映射"或"全射"。一个函数 f: A → B 是满射的，如果对于 B 中的每个元素 b，都存在 A 中的元素 a，使得 f(a) = b。简而言之，B 中的每个元素都能够由 A 中的元素映射得到。可以理解为函数的输出范围覆盖了整个目标集合 B。
3. 双射（Bijective）：也称为"一一对应映射"或"双射映射"。一个函数 f: A → B 是双射的，如果它既是单射又是满射。换句话说，对于任意 a ∈ A，存在唯一的 b ∈ B 使得 f(a) = b，并且对于任意 b ∈ B，存在唯一的 a ∈ A 使得 f(a) = b。双射可以理解为函数在输入和输出之间建立了一种完全的一一对应关系。

某个物件×成为嵌入到另一个物件Y中,是指有一个保持结构的单射f:X->Y这个映射就叫作—个嵌入

**单射:**如果x1!=x2,那么f(x1) != f(x2)
**需要保持结构︰**这个其实就是我们机器学习应用中一直探讨的东西，到底我们应该用什么样的选择去决定我们想要保持的结构

![image-20230624164553361](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230624164553361.png)

 ![image-20230624165056156](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230624165056156.png)

![image-20230624165127489](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230624165127489.png)

#### Auto Encoder

自动编码器(Autoencoder)是一种无监督的神经网络,用于学习数据的潜在表示。它通过编码输入数据到低维空间,然后解码回原始空间,来重构输入数据。

自动编码器通常包含:- 编码器(Encoder):编码输入数据到隐藏层的表示。
\- 解码器(Decoder):从隐藏层的表示解码回原始空间。
\- 潜在空间(Latent space):编码器输出的隐藏层表示,它捕获输入数据的潜在结构。

自动编码器的训练目标是重构输入数据,即最小化重构误差:

`Loss = MSE(Input, Output) = MSE(X, Reconstruction)`

其中X是输入数据,Reconstruction是解码器的输出。

自动编码器的好处在于:

1.它可以学习数据的潜在表示,这些表示反映数据的基本结构和模式。这可以用于降维,特征提取,数据压缩等。

2.重构目标使模型依赖输入数据的全部信息,这产生一个强大的正则化器,帮助学习更重要的潜在特征。

3.可以用于生成新数据,通过在潜在空间中采样和解码。

4.可以与其他模型(如分类器)连接以实现联合训练,这通常称为"自动编码器-xx"模型,例如:- 自动编码器-分类器:AE与分类器联合,可以产生更加抽象和区分性强的特征表示。

- 变分自动编码器:AE与VAE联合,以最大化数据的似然概率。这可以产生更加规范的潜在表示。
- 递归自动编码器:用于学习序列数据的潜在递归结构。自动编码器被广泛用于无监督特征学习、异常检测、图像生成等任务。它提供一个端对端的框架来学习数据的潜在特征表示,这些表示反应数据的内在结构和统计规律。

![image-20230624165809330](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230624165809330.png) 

![image-20230624165817492](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230624165817492.png)

![image-20230624170138605](C:\Users\wangzhanghang\AppData\Roaming\Typora\typora-user-images\image-20230624170138605.png)