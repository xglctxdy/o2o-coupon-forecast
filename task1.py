import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(100)
data = np.random.randint(1, 50, size=(10, 5))
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'], index=['Row{}'.format(i + 1) for i in range(10)])

# 绘制柱状图
df.plot(kind='bar')
plt.title('Bar Chart')
plt.show()

# 绘制散点图
df.plot(kind='scatter', x='A', y='B')
plt.title('Scatter Plot')
plt.show()

# 绘制饼图
df.sum().plot(kind='pie')
plt.title('Pie Chart')
plt.show()

# 绘制箱线图
df.plot(kind='box')
plt.title('Box Plot')
plt.show()

# 绘制折线图
df.plot(kind='line')
plt.title('Line Plot')
plt.show()
