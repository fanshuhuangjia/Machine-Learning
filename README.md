##[RNN](https://github.com/fanshuhuangjia/Machine-Learning/blob/master/RNN.ipynb)

* 代码简介：搭建了一个简单`RNN`模型
    * 使用tanh作为隐藏层激活函数，用softmax作为输出层激活函数 
    * 通过反向传播进行训练
* 实验过程：预测字母中的下一个字母。这里只写了前面9个字母"abcdefghi",相对需要的样本比较少。
* 实验结果：通过将学习率设为0.1，迭代次数设为100,成功给出了'f'和'abc'后的下一个字母为'g' 和'd'
![页面截图](https://github.com/fanshuhuangjia/Machine-Learning/blob/master/img-folder/RNN_result.jpg)
