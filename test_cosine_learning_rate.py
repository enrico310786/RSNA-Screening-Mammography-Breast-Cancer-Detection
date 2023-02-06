import torch
import matplotlib.pyplot as plt


model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)
lrs = []
steps = []

for i in range(30):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    steps.append(i)
    print("Step = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
    scheduler.step()


fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(steps, lrs, linewidth=2)
fig.savefig("resources/plot_learning_rate.png")