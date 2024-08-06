import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'Bedrooms': [2, 1, 2],
    'Area': [18, 6, 12],
    'Age': [25, 4, 35],
    'Price': [998, 501, 1205]
}

# plt.plot(data['Bedrooms'],label='Bedroom')
# plt.plot(data['Area'],label='Area')
# plt.plot(data['Age'],label='Age')
# plt.plot(data['Price'],label='Price')
# plt.legend()
# plt.show()

df = pd.DataFrame(data); print(df)

X = df[['Bedrooms', 'Area', 'Age']].values
y = df['Price'].values

print(X);print(y); print(y.reshape(-1,1)) # (len(y),1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# n = 0
# for i in X.T:
#     plt.plot(i,label=f'{list(data.keys())[n]}')
#     n += 1
# plt.plot(y.T[0],label='Price')
# plt.legend()
# plt.show()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

print(X); print(y)


class MLP(nn.Module):
    def __init__(self, size):
        super(MLP, self).__init__()
        self.size = size
        self.hidden = nn.Linear(3, self.size)
        self.output = nn.Linear(self.size, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


l = []
for i in range(1,21):
    model = MLP(i)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 1000
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            
    model.eval()
    with torch.no_grad():
        predicted = model(X).numpy()

    predicted = scaler_y.inverse_transform(predicted)
    print("Giá trị dự đoán:", predicted.T)

    # print(model.hidden.weight.data)
    # print(model.hidden.bias.data)
    # print(model.output.weight.data)
    # print(model.output.bias.data)
    
    l.append(losses)

plt.title('LOSS')
i = 1
for lx in l:
    plt.plot(lx,label=f'{i}')
    i += 1
plt.legend()
plt.show()