import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ..logger import logger

class FaceJudgeModel(nn.Module):
    """Simple binary CNN classifying images as Face vs. Non-Face."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_judge(real_images, device="cuda"):
    """Trains the Face vs. Non-Face classifier on real CelebA faces and negative noise targets."""
    logger.info("Training the Face Judge classifier...")
    model = FaceJudgeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. Prepare positive samples (Real CelebA)
    pos_x = real_images[:1000]
    pos_y = torch.ones(len(pos_x), dtype=torch.long)
    
    # 2. Prepare negative samples (structured noise and random shapes)
    neg_x = torch.randn_like(pos_x) * 0.5
    # Add some solid patterns to negatives to make it harder
    for i in range(len(neg_x) // 2):
        neg_x[i, :, :, random_idx := torch.randint(0, 32, (1,))] = 1.0
    neg_y = torch.zeros(len(neg_x), dtype=torch.long)
    
    # 3. Create loader
    x = torch.cat([pos_x, neg_x], dim=0)
    y = torch.cat([pos_y, neg_y], dim=0)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 4. Train loop (5 epochs is plenty for 99% accuracy)
    model.train()
    criterion = nn.CrossEntropyLoss()
    for _ in range(5):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    logger.info("Face Judge training completed.")
    return model

@torch.no_grad()
def get_legibility_score(judge, gen_images, device="cuda"):
    """Evaluates the Legibility Score: mean probability of being classified as a face."""
    judge.eval()
    batch = gen_images.to(device)
    # Output logits
    logits = judge(batch)
    probs = F.softmax(logits, dim=1)
    # Mean confidence of index 1 (Face class)
    mean_conf = probs[:, 1].mean().item()
    return mean_conf
