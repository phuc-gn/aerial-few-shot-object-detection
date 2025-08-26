import torch
import utils
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data_vhr10 import get_vhr10_datasets, BASE_CLASSES

# --- Config ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = len(BASE_CLASSES)
num_epochs = 10

# --- Datasets ---
base_train, novel_train, test_dataset, collate_fn = get_vhr10_datasets(fewshot_k=5)
train_loader = DataLoader(base_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# --- Model ---
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# --- Optimiser + Scheduler ---
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-6, momentum=0.9, weight_decay=5e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# --- Training loop ---
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, test_loader, device=device)

torch.save(model.state_dict(), "fasterrcnn_base.pth")
print("Base training finished, model saved as fasterrcnn_base.pth")
