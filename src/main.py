import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(torch.get_num_threads())
print("Потоков используется:", torch.get_num_threads())

# 1. Данные: y = 2x + 1 с шумом
torch.manual_seed(0)
x1 = 2 * torch.rand(1000, 1) - 1
x2 = 2 * torch.rand(1000, 1) - 1
x = torch.cat([x1, x2], dim=1)
y = (((x1**2 + x2**2)**0.5)>=0.5).float()

# 2. Определяем модель
model = nn.Sequential(
	nn.Linear(2, 100),
	nn.Tanh(),
	nn.Linear(100, 200),
	nn.Tanh(),
	nn.Linear(200, 100),
	nn.Tanh(),
	nn.Linear(100, 1),
	nn.Sigmoid()
)

# 3. Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Обучение
epochs = 10_000
for epoch in range(epochs):
    # прямой проход
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # обнуление градиентов и обратное распространение
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 40 == 0:
        print(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')


torch.save(model.state_dict(), "model.pth")
# 5. Проверка результата
tests = 100
test_x1 = torch.unsqueeze(torch.linspace(-1, 1, tests), dim=1)
test_x2 = torch.unsqueeze(torch.linspace(-1, 1, tests), dim=1)
test_y = (((test_x1**2 + test_x2**2)**0.5) >= 0.5).float()

for i in range(tests):
    # объединяем по dim=0, т.к. каждый x[i] имеет форму [1]
    test_x = torch.cat([test_x1[i], test_x2[i]], dim=0).unsqueeze(0)  # форма [1,2]
    pred_y = model(test_x)
    print(f"Нейросеть предсказала: {pred_y.item():.3f}; правильный ответ: {test_y[i].item():.0f}; точка: ({test_x})")


