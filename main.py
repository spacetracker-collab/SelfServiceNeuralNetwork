"""
Neural Self-Service Pattern Model
Based on: CBOACR (Customer, Behavior, Objective, Aid, Channel, Relationship)

This model learns:
Input  -> Context Features
Output -> Multi-head predictions for pattern components

Author-inspired implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIM = 32

# Pattern heads (from paper taxonomy)
NUM_BEHAVIOR = 6
NUM_AID = 6
NUM_CUSTOMER = 4
NUM_CHANNEL = 4
NUM_RELATION = 4

HIDDEN_DIM = 128
LR = 0.001
EPOCHS = 25


# -----------------------------
# MODEL
# -----------------------------
class SelfServiceNet(nn.Module):
    def __init__(self):
        super(SelfServiceNet, self).__init__()

        # Shared encoder (captures context)
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )

        # Multi-head outputs (Pattern Language Heads)
        self.behavior_head = nn.Linear(HIDDEN_DIM, NUM_BEHAVIOR)
        self.aid_head = nn.Linear(HIDDEN_DIM, NUM_AID)
        self.customer_head = nn.Linear(HIDDEN_DIM, NUM_CUSTOMER)
        self.channel_head = nn.Linear(HIDDEN_DIM, NUM_CHANNEL)
        self.relation_head = nn.Linear(HIDDEN_DIM, NUM_RELATION)

    def forward(self, x):
        h = self.encoder(x)

        return {
            "behavior": self.behavior_head(h),
            "aid": self.aid_head(h),
            "customer": self.customer_head(h),
            "channel": self.channel_head(h),
            "relation": self.relation_head(h),
        }


# -----------------------------
# LOSS FUNCTION
# -----------------------------
def compute_loss(outputs, targets):
    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    loss += loss_fn(outputs["behavior"], targets["behavior"])
    loss += loss_fn(outputs["aid"], targets["aid"])
    loss += loss_fn(outputs["customer"], targets["customer"])
    loss += loss_fn(outputs["channel"], targets["channel"])
    loss += loss_fn(outputs["relation"], targets["relation"])

    return loss


# -----------------------------
# SYNTHETIC DATA GENERATOR
# -----------------------------
def generate_data(batch_size=64):
    x = torch.randn(batch_size, INPUT_DIM)

    targets = {
        "behavior": torch.randint(0, NUM_BEHAVIOR, (batch_size,)),
        "aid": torch.randint(0, NUM_AID, (batch_size,)),
        "customer": torch.randint(0, NUM_CUSTOMER, (batch_size,)),
        "channel": torch.randint(0, NUM_CHANNEL, (batch_size,)),
        "relation": torch.randint(0, NUM_RELATION, (batch_size,))
    }

    return x, targets


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train():
    model = SelfServiceNet()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        x, targets = generate_data()

        outputs = model(x)
        loss = compute_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss {loss.item():.4f}")

    return model


# -----------------------------
# INFERENCE
# -----------------------------
def infer(model):
    x = torch.randn(1, INPUT_DIM)
    outputs = model(x)

    predictions = {k: torch.argmax(v, dim=1).item() for k, v in outputs.items()}

    print("\n--- Inference Result ---")
    print(predictions)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    model = train()
    infer(model)
