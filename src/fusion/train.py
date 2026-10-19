import copy

import numpy as np
import torch
from torch.utils.data import DataLoader


def train_model(model, train_ds, val_ds, device, epochs=30, patience=5,
                lr=1e-3, batch_size=256, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = torch.nn.MSELoss()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    best_val, best_state, bad = np.inf, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for z_ts, z_llm, y in train_dl:
            z_ts, z_llm, y = z_ts.to(device), z_llm.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            pred, _ = model(z_ts, z_llm)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        train_loss = tot / n

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for z_ts, z_llm, y in val_dl:
                z_ts, z_llm, y = z_ts.to(device), z_llm.to(device), y.to(device).unsqueeze(1)
                pred, _ = model(z_ts, z_llm)
                v = crit(pred, y).item()
                vtot += v * len(y)
                vn += len(y)
        val_loss = vtot / vn
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        if val_loss < best_val - 1e-7:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
