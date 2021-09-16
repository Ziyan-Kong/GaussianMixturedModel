setwd("D:\\worktask\\20210914001_Gaussian")
# 安装包
install.packages("mclust")
install.packages("gclus")

# 加载库
library(mclust)
library(gclus)
library(ggplot2)

# 加载数据
data("wine", package = "gclus")

# 第一列和聚类没有关系，删除
x <- data.matrix(wine[,-1])
mod <- Mclust(x)

# 输出聚类结果与已知结果的比较
table(wine$Class, mod$classification)

# 评估聚类效果
adjustedRandIndex(wine$Class, mod$classification)

# 基于BIC准则，确定聚类数目
# Mclucst会选择其中BIC最大的模型和分组作为最终的结果
plot.Mclust(mod, what = "BIC", 
            ylim = range(mod$BIC[,-(1:2)], na.rm = TRUE), 
            legendArgs = list(x = "bottomleft", cex =0.7))


drmod <- MclustDR(mod, lambda = 1)
plot(drmod)

# real data
load(file = "CorData.RData")

dat <- CorData[,c(1,3)]
colnames(dat) <- c("CNV_Protein", "CNV_mRNA")
dat <- dat[complete.cases(dat),]
atten <- dat$CNV_mRNA - dat$CNV_Protein


mod2 <- Mclust(atten, G = 2, modelNames = "E")
  
  
dat$Class <- as.factor(mod2$classification)
  
ggplot(data = dat, aes(y = CNV_mRNA, x = CNV_Protein, color = Class)) +
    geom_point() +
    scale_x_continuous(
      limits = c(-1, 1)
    ) +
    scale_y_continuous(
      limits = c(-1, 1)
    )


drmod2 <- MclustDR(mod2)
plot(drmod2)


