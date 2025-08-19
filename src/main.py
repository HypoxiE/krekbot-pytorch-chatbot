import torch
import torch.nn as nn
import torch.optim as optim

# 1. Данные: y = 2x + 1 с шумом
torch.manual_seed(0)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # shape: [100,1]
y = 2 * x + 1 + 0.2 * torch.randn(x.size())

# 2. Определяем модель (1 скрытый слой, 10 нейронов)
model = nn.Sequential(
    nn.Linear(1, 10),  # вход -> скрытый слой
    nn.ReLU(),         # активация
    nn.Linear(10, 1)   # скрытый -> выход
)

# 3. Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Обучение
for epoch in range(200):
    # прямой проход
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # обнуление градиентов и обратное распространение
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 40 == 0:
        print(f'Epoch {epoch+1}/200 | Loss: {loss.item():.4f}')

# 5. Проверка результата
test_x = torch.tensor([[0.5]])
pred_y = model(test_x)
print(f"При x=0.5 сеть предсказывает: {pred_y.item():.3f}")
