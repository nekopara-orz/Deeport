#对DeepSort代码中Kalman滤波算法做详细解读
现阶段网上所有对Kalman滤波代码的的讲解普遍很浅显，部分讲的深入一点文章对代码解析的也不够细致，并没有深入到具体的某一行代码具体的目的是什么。由于原作者有些地方写的很巧妙不做一些详细的注释会一时难以看懂。因此我将对KalmanFilter类中的代码进行详细说明。
>本文假设读者已经对卡尔曼滤波算法有了一定的了解，如果尚未了解该算法推荐使用下面链接的文章进行学习
> 英文原版: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/ \
> 中文翻译版: https://zhuanlan.zhihu.com/p/39912633 \
##1.初始化参数
在运算前要对一些参数进行初始化一些参如协方差矩阵
```latex
f(x) = \int_{-\infty}^\infty
    \hat f(\xi)\,e^{2 \pi i \xi x}
    \,d\xi
```
