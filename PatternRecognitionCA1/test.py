import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sympy import false


X1=np.load('class1.npy') #读入第一类数据
X2=np.load('class2.npy') #读入第二类数据
X_concat = np.concatenate((X1, X2)) # 380行2列的矩阵
Y_concat = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))  # 创建标签，第一类真值为0，第二类真值为1，shape[0]返回矩阵的行数，也就是数据点的个数，一类数据300个，二类数据80个
# 转换为 PyTorch Tensor
X_tensor = torch.tensor(X_concat, dtype=torch.float32)
Y_tensor = torch.tensor(Y_concat, dtype=torch.float32).view(-1, 1)  # 变为列向量


class MLP(nn.Module): #定义一个MLP类，继承自torch.nn.Module，提供了管理参数/模型结构和前向传播的功能
    def __init__(self): #类构造函数
        super(MLP,self).__init__() #对nn.Module初始化，调用torch.nn.Module的构造函数__init__()，确保MLP继承自nn.Module的所有功能
        self.hidden = nn.Linear(2,3) #输入两个神经元，隐藏层三个神经元
        self.out = nn.Linear(3,1) #隐藏层三个神经元，输出一个神经元
        self.relu = nn.ReLU() #隐藏层后激活函数选用ReLU
        self.sigmoid = nn.Sigmoid() #输出层后应用Sigmoid解决二分类问题

        nn.init.normal_(self.hidden.weight, mean=0.0, std=0.1) #遵循正态分布初始化隐藏层前的权重
        nn.init.normal_(self.out.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.hidden.bias) #初始偏置预设为0
        nn.init.zeros_(self.out.bias)

    def forward(self,x):
        z1 = self.hidden(x)
        h1 = self.relu(z1) #隐藏层的前向传播
        z2 = self.out(h1)
        h2 = self.sigmoid(z2) #输出层的前向传播
        return z1, h1, z2, h2 #出来的矩阵维度为380，1，其中1是指分类概率


mlp=MLP()
epochs = 30000
learning_rate = 0.1


def relu_derivative(x):
    return (x > 0).float()


def sigmoid_derivative(x):
    sig = torch.sigmoid(x)
    return sig*(1-sig)


def back_propagation(model, x, y, lr):
    z1, h1, z2, h2 = model(x)
    y_predict = h2
    loss_value = torch.mean(0.5 * (y_predict-y) ** 2)
    delta_out = (y_predict-y) * sigmoid_derivative(z2)
    dw2= h1.T @ delta_out / len(x)
    delta_hidden = delta_out @ model.out.weight * relu_derivative(z1)
    dw1 = x.T @ delta_hidden / len(x)
    db2 = torch.mean(delta_out, dim=0)
    db1 = torch.mean(delta_hidden, dim=0)

    # 更新权重
    model.out.weight.data -= lr * dw2.T
    model.out.bias.data -= lr * db2

    model.hidden.weight.data -= lr * dw1.T
    model.hidden.bias.data -= lr * db1

    return loss_value.item(), model.hidden.weight.data, model.out.weight.data


for epoch in range(epochs):
    loss, w1, w2, = back_propagation(mlp, X_tensor, Y_tensor, learning_rate)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')


x_min, x_max = X_concat[:, 0].min() - 1, X_concat[:, 0].max()+1 # X 轴范围
y_min, y_max = X_concat[:, 1].min() - 1, X_concat[:, 1].max()+1  # Y 轴范围
res = 0.01  # 网格间隔
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))


# 将网格点转换为 PyTorch 张量
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32) #ravel将二维网格展平成一维数组，np_c再将其拼接成二维坐标的数组
grid_tensor = torch.tensor(grid_points) #将二维数组转换为张量


#计算 MLP 预测（未训练）
_, _, _, h_2 = mlp(grid_tensor)
Z = h_2.detach().numpy()  # 预测输出，最后将张量转换为数组，此时Z变为一个列向量，里面的数字代表概率
Z = Z.reshape(xx.shape)  # 调整形状以匹配网格，从一维数组变成了二维的矩阵，数量和矩阵的点数相同


# 绘制决策边界
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm_r", alpha=0.3)
# coutourf用于绘制等高线图，通过颜色的不同表示不同区域的分类，level定义了等高线的级别，即颜色区间的切分点，第一类和第二类的分界线为0.5
# coolwarm代表用冷暖色点映射颜色，alpha=0.3表示透明度，让等高线图的颜色更加透明

# 绘制数据点
plt.scatter(X1[:, 0], X1[:, 1], color='red', label="Class 1")
plt.scatter(X2[:, 0], X2[:, 1], color='blue', label="Class 2")

plt.legend() #添加图例
plt.title("MLP Decision Boundary (trained)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show(block=false)

# 计算并输出性能指标（仅对 X_concat 数据进行推理）

X_concat_tensor = torch.tensor(X_concat, dtype=torch.float32)
z__1, h__1, z__2, h__2 = mlp(X_concat_tensor)
Y_pred= h__2.detach().numpy()  # 获取预测值
Y_pred_class = (Y_pred > 0.5).astype(int)  # 将概率转换为分类


# 计算分类指标
accuracy = accuracy_score(Y_concat, Y_pred_class) #准确率
precision = precision_score(Y_concat, Y_pred_class, zero_division=1)
recall = recall_score(Y_concat, Y_pred_class, zero_division=1)

# 输出指标
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# 计算并绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(Y_concat, Y_pred)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
