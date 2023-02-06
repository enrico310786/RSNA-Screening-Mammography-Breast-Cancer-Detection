import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pos_class_weight = 46
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_class_weight))

y_true = torch.tensor([1,0,1,0]).unsqueeze(1).float()
print("y_true.size()", y_true.size())
y_logit = torch.tensor([1.00, 1.00, 1.00, -1.00]).unsqueeze(1)
print("y_logit.size()", y_logit.size())


loss_value = criterion(y_logit, y_true).item()
print("loss_value: ", loss_value)