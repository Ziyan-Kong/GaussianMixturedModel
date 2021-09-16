
# 安装包
install.packages("mclust")
install.packages("gclus")

# 加载库
library(mclust)
library(gclus)
library(ggplot2)

# ------------------ 总体预览 ------------------ #
# 加载数据
data("wine", package = "gclus")

# 第一列不参与聚类，删除
x <- data.matrix(wine[,-1])
mod <- Mclust(x)

# 输出聚类结果与已知结果的比较,类似于混淆矩阵
table(wine$Class, mod$classification)

# 评估聚类效果
adjustedRandIndex(wine$Class, mod$classification)

# -------------------- 调参 ----------------------- #
# Mclust如何挑选模型以及它为什么认为聚成3类比较合适呢？
# 我们可以根据什么信息进行模型选择呢？
# step 1.模型选择-主要针对采用什么形式的方差类型
# 为了解答上面的问题，我们需要稍微了解点Mclust的原理。和其他基于模型的方法类似，
# Mclust假设观测数据是一个或多个混合高斯分布的抽样结果，
# Mclust就需要根据现有数据去推断最优可能的模型参数，以及是由几组分布抽样而成。
# mclust一共提供了14种模型，可以用?mclustModelNames可以查看mclust提供的所有模型。


# 和问题直接相关的是如下两个参数
# G: 分组数，默认情况下是1:9
# modelNames: 待拟合的模型，默认使用所有14种。
# 默认情况下，Mclust得到14种模型中1到9组的分析结果，然后根据一定的标准选择最终的模型和分组数

# Step 2. 确定最佳分类数量
# Mclust提供了两种方法用于评估不同模型在不同分组下的可能性
# BIC( Bayesian Information Criterion ): 贝叶斯信息判别标准
# ICL( integrated complete-data likelihood ): 综合完全数据可能性
# Mclust默认用的就是BIC，因此我们可以用plot.Mclust绘制其中BIC变化曲线

# 基于BIC准则，确定聚类数目
# Mclucst会选择其中BIC最大的模型和分组作为最终的结果
plot.Mclust(mod, what = "BIC", 
            ylim = range(mod$BIC[,-(1:2)], na.rm = TRUE), 
            legendArgs = list(x = "bottomleft", cex =0.7))

# 此外, 我们可以用MclustBIC和MclustICL分别进行计算
# 从中选择最佳的模型分组和模型作为输入
par(mfrow=c(1,2))
BIC <- mclustBIC(X)
ICL <- mclustICL(X)

mod2 <- Mclust(X, G = 3, modelNames = "VVE", x=BIC)

# ---------------- 可视化 ----------------------- #
# mclust为不同的输出都提供了对应的泛型函数用于可视化，
# 你需要用plot就能得到结果。
# 例如对之前的聚类结果在二维空间展示
drmod <- MclustDR(mod, lambda = 1)
plot(drmod)

# ----------------- real data ---------------- #
load(file = "CorData.RData")

dat <- CorData[,c(1,3)]
colnames(dat) <- c("CNV_Protein", "CNV_mRNA")
dat <- dat[complete.cases(dat),]
atten <- dat$CNV_mRNA - dat$CNV_Protein


mod2 <- Mclust(atten, G = 2, modelNames = "E")
  
  
dat$Class <- as.factor(mod2$classification)
dat$Group <- ifelse(dat$Class==1, "High", "Low")
  
ggplot(data = dat, aes(y = CNV_mRNA, x = CNV_Protein, color = Group)) +
    geom_point() +
    stat_density2d(data = dat, aes(y = .data$CNV_mRNA, x = .data$CNV_Protein, group=.data$Class),
                   color = "black", lwd = 1) +
    scale_x_continuous(
      limits = c(-1, 1)
    ) +
    scale_y_continuous(
      limits = c(-1, 1)
    ) +
  scale_color_manual(
    values = c("red", "blue")
  )+
  theme_linedraw()


drmod2 <- MclustDR(mod2)
plot(drmod2)

# reference
# mclust 5: Clustering, Classification and Density Estimation Using Gaussian Finite Mixture Models
# https://www.biotechknowledgestudy.com/802.html

