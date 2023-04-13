import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

class Config():
    data_path = '../data/stock_dataset.csv'
    timestep = 1
    batch_size = 32
    feature_size = 1
    hidden_size = 256
    output_size = 1
    num_layers = 2
    epochs = 10
    best_loss = 0
    learning_rate = 0.0003
    model_name = 'rnn'
    save_path = './{}.pth'.format(model_name)

config = Config()


# 1.Loading the Stock Data
pro = ts.pro_api('your token')
df = pro.daily(ts_code='000001.SZ', start_date='20130711', end_date='20220711')
df.index = pd.to_datetime(df.trade_date)
df = df.iloc[::-1]

# 2.Standardization
scaler = StandardScaler()
scaler_model = StandardScaler()
data = scaler_model.fit_transform(np.array(df[['open', 'high', 'low', 'close']]).reshape(-1, 4))
scaler.fit_transform(np.array(df['close']).reshape(-1, 1))


# 3. Split Data
def split_data(data, timestep):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep])
        dataY.append(data[index + timestep][3])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, 4)
    y_train = dataY[: train_size]

    x_test = dataX[train_size:, :].reshape(-1, timestep, 4)
    y_test = dataY[train_size:]

    return [x_train, y_train, x_test, y_test]


# 3.
x_train, y_train, x_test, y_test = split_data(data, config.timestep)

# 4. Transform the dataset into tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 5.Generate the training set
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.Data Loader
train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, True)
test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, False)

# 7.RNN
class RNN(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  #
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.rnn = nn.RNN(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden

        # RNN运算
        output, h_n = self.rnn(x, h_0)

        # 全连接层
        output = self.fc(output)  # 形状为 batch_size, timestep, output_size

        return output[:, -1, :]

model = RNN(config.feature_size, config.hidden_size, config.num_layers, config.output_size)  # 定义RNN网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 定义优化器

# 8.Traning the Model
for epoch in range(config.epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, config.epochs, loss)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), config.save_path)

print('Finished Training')

# 9.绘制结果
plot_size = 200
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1)), "b")
plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
plt.legend()
plt.show()

y_test_pred = model(x_test_tensor)
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[: plot_size]), "b")
plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
plt.legend()
plt.show()
