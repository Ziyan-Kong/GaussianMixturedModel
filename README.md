# Gaussian Mixtured Model 

# 一、简介

无监督学习相比于监督学习有着更大的挑战空间，因为我们不知道我们的新数据存在哪些趋势或模式，这个给我们最后调参或者选择方法有了更大的挑战。

在无监督学习中，最流行的技术之一是聚类，可用于疾病患者的分层，消费者分层，市场购物篮分析等。

这里我们主要来讲解一下基于高斯混合模型的概率估算器，并将该模型与常用的K-means算法进行比较，以突出高斯混合模型的优势。

# 二、高斯混合模型概念

`混合模型` 是一个可以用来表示在总体分布（distribution）中含有 K 个子分布的概率模型，换句话说，混合模型表示了观测数据在总体中的概率分布，它是一个由 K 个子分布组成的混合分布。混合模型能够计算观测数据在总体分布中的概率，并依据概率值判定属于哪一种子分布。

---

`单维高斯分布密度函数`：![image-20210915173555196](../../software/Typora/image/image-20210915173555196.png)

---

`多维高斯分布密度函数`：![image-20210915173634462](../../software/Typora/image/image-20210915173634462.png)



其中，μ为数据均值（期望值），ε为协方差，k为数据的维度。

---

`混合高斯分布`：多个高斯分布的加权平均（如下）。

![image-20210916091251270](../../software/Typora/image/image-20210916091251270.png)

其中，k表示（1...K）个子分布。

一般采用基于EM算法进行求解：

- 初始化参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) , ![[公式]](https://www.zhihu.com/equation?tex=%5CSigma) , ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) ,K(高斯分布的数目)
- 计算出后验概率
- 更新参数值
- 重复2-3步直到收敛

# 三、高斯混合模型与K-Means的比较

## 3.1 K-Means的计算

k-means是一种基于距离的模式聚类，其具体的步骤为：

- 基于给定的k个聚类数量，随机初始化k个质心，基于距离算法将数据点指定给最近的质心，形成一个簇；
- 随后，基于形成的簇更新质心，重新分配数据点；
- 重复上述两个步骤，直到质心不在变化。

```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

# Plot the data with K Means Labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
```

![image-20210916094632468](../../software/Typora/image/image-20210916094632468.png)

基于KMeans的模式聚类，某种意义上给出了每一个样本点的确定性的簇。但是，有些点确实存在一些不确定性的，比如中间两个类别间的数据点存在轻微的重叠，我们对于这种分配是没有完全的信心。而k-means模式聚类不存在分配概率或不确定性的度量。

k-means模型总体思路可以理解为它在每个类蔟的中心放置了一个圆（或者更高维度超球面)，其半径由聚类中最远的点确定。该半径充当训练集中聚类分配的一个硬截断：任何圆外的数据点不被视为该类的成员。我们可以使用以下函数可视化这个聚类模型：

```python
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)
```

![image-20210916095634313](../../software/Typora/image/image-20210916095634313.png)

基于上述的计算结果，我们可以看出这些聚类模式是一个圆形。k-means没有内置的方法计算椭圆形或椭圆形的簇。因此，假设我们将相同的数据点做变化，这种聚类模式将会变得比较混乱。

```python
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)
```

![image-20210916095943927](../../software/Typora/image/image-20210916095943927.png)

总之，基于k-means的模式聚类存在两个明显的缺陷：聚类形状不够灵活；缺少聚类分配的概率值。

## 3.2 高斯混合模型

### 3.2.1 概述

---

在一维空间中（仅有一个变量），高斯分布是一个钟形曲线，数据点围绕平均值对称分布。

![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/heS6wRSHVMk9MK63JSapmYUKicz9KELT5Cbw0dS6UOdPo0h9lLgAMGv3sJkY92UOyybuc6HDGAibialcHec9ZEiaug/640?wx_fmt=jpeg)

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/heS6wRSHVMmrUfLl2UooTw3v3pytwiaWxcibGFMNfN5lIqj9l5JU9agofUI50byLskwEFsOQ5sLX7C2TmNdticDdQ/640?wx_fmt=png)

---

在多维空间中（如存在两个变量时），我们将得到一个三维钟形曲线。

![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/heS6wRSHVMmrUfLl2UooTw3v3pytwiaWxyuUmv6SgRkcYWIDKiav8I8cRoSfUaqYCicVMn5ESDTyDecNNiaQd5h8Zg/640?wx_fmt=png)

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/heS6wRSHVMmrUfLl2UooTw3v3pytwiaWxgOx1WrmuSuXaibXmOqFl65KGNbL4oDBLOnAyzk5hL8icyIwtbPsHsjNQ/640?wx_fmt=png)

其中，x是输入向量，μ是2维均值向量，∑是2×2协方差矩阵。协方差现在可以决定曲线的形状。d维概率密度函数可以类似进行推广。

**“因此，这个多元高斯模型将x和μ作为长度为d的向量，∑是一个d×d协方差矩阵。”**

对于具有d个特征的数据集，我们将得到k个高斯分布（其中k相当于簇的数量），每个高斯分布都有一个特定的均值向量和方差矩阵，但是——这些高斯分布的均值和方差值是如何给定的？这些值可以用一种叫做期望最大化（Expectation-Maximization ，EM）的技术来确定。

### 3.2.2 基于EM的高斯混合模型

高斯混合模型（GMM）试图找到一个多维高斯概率分布的混合，以模拟任何输入数据集。在最简单的情况下，GMM可用于以与k-means相同的方式聚类。

```python
from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
```

![image-20210916103124858](../../software/Typora/image/image-20210916103124858.png)

但因为GMM包含概率模型，因此可以找到聚类分配的概率方式 。在Scikit-Learn中，通过调用predict_proba方法实现。它将返回一个大小为[n_samples, n_clusters]的矩阵，用于衡量每个点属于给定类别的概率：

```python
probs = gmm.predict_proba(X)
print(probs[:5].round(3))
```

```markdown
[[0.537 0.463 0.    0.   ]
 [0.    0.    1.    0.   ]
 [0.    0.    1.    0.   ]
 [1.    0.    0.    0.   ]
 [0.    0.    1.    0.   ]]
```

我们可以可视化这种不确定性，比如每个点的大小与预测的确定性成比例；如下图，我们可以看到正是群集之间边界处的点反映了群集分配的不确定性：

```python
size = 50 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size);
```



![image-20210916103533816](../../software/Typora/image/image-20210916103533816.png)



本质上说，高斯混合模型与k-means非常相似：它使用期望-最大化的方式，定性地执行以下操作：

1. 选择位置和形状的初始猜想
2. 重复直到收敛
   1. E步骤：对于每个点，计算其属于每个类别的概率权重
   2. M步骤：对于每个类别，使用E步算出的权重，根据所有数据点，更新其位置，规范化和形状 

`结果是，每个类别不是被硬边界的球体界定，而是平滑的高斯模型。正如在k-means的期望-最大方法一样，这种算法有时可能会错过全局最优解，因此在实践中使用多个随机初始化`。

 让我们创建一个函数，通过基于GMM输出，绘制椭圆来帮助我们可视化GMM聚类的位置和形状：

```python
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

gmm = GMM(n_components=4, random_state=42)
plot_gmm(gmm, X)
```

![image-20210916104033614](../../software/Typora/image/image-20210916104033614.png)

同样，我们可以使用GMM方法来拟合我们的拉伸数据集；允许full的协方差，该模型甚至可以适应非常椭圆形，伸展的聚类模式：

```python
gmm = GMM(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)
```

![image-20210916104138985](../../software/Typora/image/image-20210916104138985.png)

这两个实列表明GMM能够解决k-means解决不了的问题。

> 高斯分布的协方差选择
>
> 如果看了之前拟合的细节，你将看到covariance_type选项在每个中都设置不同。该超参数控制每个类簇的形状的自由度；对于任意给定的问题，必须仔细设置。默认值为covariance_type =“diag”，这意味着可以独立设置沿每个维度的类蔟大小，并将得到的椭圆约束为与轴对齐。一个稍微简单和快速的模型是covariance_type =“spherical”，它约束了类簇的形状，使得所有维度都相等。尽管它并不完全等效，其产生的聚类将具有与k均值相似的特征。更复杂且计算量更大的模型（特别是随着维数的增长）是使用covariance_type =“full”，这允许将每个簇建模为具有任意方向的椭圆。
>  对于一个类蔟，下图我们可以看到这三个选项的可视化表示：
>
> ![img](https://upload-images.jianshu.io/upload_images/9896155-162fae0b3d9a8965.jpg?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

### 3.2.3 混合高斯分布的密度估计解说

尽管GMM通常被归类为聚类算法，但从根本上说它是一种密度估算算法。也就是说，GMM适合某些数据的结果在技术上不是聚类模型，而是描述数据分布的生成概率模型。

例如，考虑一下Scikit-Learn的make_moons函数生成的一些数据：

```python
from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1]);
```

![image-20210916104750694](../../software/Typora/image/image-20210916104750694.png)

如果我们尝试用视为聚类模型的双成分的GMM模拟数据，则结果不是特别有用：

```python
gmm2 = GMM(n_components=2, covariance_type='full', random_state=0)
plot_gmm(gmm2, Xmoon)
```

![image-20210916104838382](../../software/Typora/image/image-20210916104838382.png)

但是如果我们使用更多成分的GMM模型，并忽视聚类的类别，我们会发现更接近输入数据的拟合：

```python
gmm16 = GMM(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, Xmoon, label=False)
```

![image-20210916104927105](../../software/Typora/image/image-20210916104927105.png)

这里，16个高斯分布的混合不是为了找到分离的数据簇，而是为了对输入数据的整体分布进行建模。这是数据分布的一个生成模型，这意味着GMM为我们提供了生成与我们的输入类似分布的新随机数据的方法。例如，以下是从这个16分量GMM拟合到我们原始数据的400个新点：

```python
Xnew = gmm16.sample(400)
plt.scatter(Xnew[0][:, 0], Xnew[0][:, 1]);
```

![image-20210916105443657](../../software/Typora/image/image-20210916105443657.png)

GMM非常方便，可以灵活地建模任意多维数据分布。

### 3.2.4 确定多少个components

GMM是一种生成模型这一事实为我们提供了一种确定给定数据集的最佳组件数的自然方法。生成模型本质上是数据集的概率分布，因此我们可以简单地评估模型下数据的可能性，使用交叉验证来避免过度拟合。校正过度拟合的另一种方法是使用一些分析标准来调整模型可能性，例如[Akaike information criterion (AIC)](https://en.wikipedia.org/wiki/Akaike_information_criterion) 或 [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion)。Scikit-Learn的GMM估计器实际上包含计算这两者的内置方法，因此在这种方法上操作非常容易。
 让我们看看在moon数据集中，使用AIC和BIC函数确定GMM组件数量：

```python
n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(Xmoon)
          for n in n_components]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
```

![image-20210916105741828](../../software/Typora/image/image-20210916105741828.png)

最佳的聚类数目是使得AIC或BIC最小化的值，具体取决于我们希望使用的近似值。 AIC告诉我们，我们上面选择的16个组件可能太多了：大约8-12个组件可能是更好的选择。与此类问题一样，BIC建议使用更简单的模型。
 注意重点：这个组件数量的选择衡量GMM作为密度估算器的效果，而不是它作为聚类算法的效果。我鼓励您将GMM主要视为密度估算器，并且只有在简单数据集中保证时才将其用于聚类。

