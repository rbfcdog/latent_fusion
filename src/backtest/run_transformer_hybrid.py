import sys, os, time, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

_root = Path(os.getcwd())
while not ((_root / 'src').exists() and (_root / 'pyproject.toml').exists()) and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))
warnings.filterwarnings('ignore')

from src.models.transformer_ts import TimeSeriesTransformer
import torch.nn as nn
import torch.optim as optim

def train_eval_transformer(model, train_loader, test_loader, epochs=30, lr=1e-3, device="cpu"):
    device = torch.device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1, :]
            tgt_output = tgt[:, 1:, :]
            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)
        
        model.eval()
        total_test = 0.0
        with torch.no_grad():
            for src, tgt in test_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1, :]
                tgt_output = tgt[:, 1:, :]
                output = model(src, tgt_input)
                loss = criterion(output, tgt_output)
                total_test += loss.item()
        avg_test = total_test / max(len(test_loader), 1)
        test_losses.append(avg_test)
        
        scheduler.step()
        
    return train_losses, test_losses

def evaluate_portfolio(model, src_tensor, actual_returns, device="cpu"):
    model.eval()
    with torch.no_grad():
        preds = model.predict(src_tensor.to(device), steps=1)[:, 0, :].cpu().numpy()
    
    # Softmax for long-only allocation
    exp_preds = np.exp(preds * 10) # scale to sharpen distribution
    weights = exp_preds / exp_preds.sum(axis=1, keepdims=True)
    
    port_returns = (weights * actual_returns).sum(axis=1)
    bh_returns = actual_returns.mean(axis=1)
    
    port_cum = np.cumprod(1 + port_returns)
    bh_cum = np.cumprod(1 + bh_returns)
    
    sharpe = np.mean(port_returns) / (np.std(port_returns) + 1e-8) * np.sqrt(252)
    
    return {
        'return_pct': (port_cum[-1] - 1) * 100,
        'bh_return_pct': (bh_cum[-1] - 1) * 100,
        'sharpe': sharpe,
        'weights': weights,
        'port_returns': port_returns
    }

def main():
    print("Loading hybrid dataset...")
    df = pd.read_parquet(_root / 'data/lse_market_data/combined_1d.parquet')
    
    # Pivot to get a clean multivariate time series
    pivot = df.pivot_table(index='timestamp', columns='symbol', values='close')
    # Filter tickers with at least 300 days of data
    valid_cols = pivot.columns[pivot.count() >= 300]
    pivot = pivot[valid_cols]
    pivot = pivot.ffill().bfill()
    
    returns = pivot.pct_change().fillna(0)
    std = returns.std() + 1e-8
    returns_norm = (returns - returns.mean()) / std
    
    print(f"Data shape: {returns.shape}, Tickers: {len(valid_cols)}")
    
    X = returns_norm.values
    seq_len = 20
    pred_len = 5
    
    src_list, tgt_list = [], []
    ret_tgt_list = []
    
    for i in range(len(X) - seq_len - pred_len):
        src_list.append(X[i : i+seq_len])
        tgt_list.append(X[i+seq_len-1 : i+seq_len+pred_len])
        # Actual returns to evaluate on T+1
        ret_tgt_list.append(returns.values[i+seq_len])
        
    src_tensor = torch.tensor(np.array(src_list), dtype=torch.float32)
    tgt_tensor = torch.tensor(np.array(tgt_list), dtype=torch.float32)
    actual_returns = np.array(ret_tgt_list)
    
    split_idx = int(len(src_tensor) * 0.7)
    
    src_train, tgt_train = src_tensor[:split_idx], tgt_tensor[:split_idx]
    src_test, tgt_test = src_tensor[split_idx:], tgt_tensor[split_idx:]
    ret_train = actual_returns[:split_idx]
    ret_test = actual_returns[split_idx:]
    
    train_ds = TensorDataset(src_train, tgt_train)
    test_ds = TensorDataset(src_test, tgt_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    input_dim = X.shape[1]
    
    print("\n=== TEST 1: Train vs Test Gap ===")
    model = TimeSeriesTransformer(input_dim=input_dim, d_model=64, n_heads=4, 
                                  n_encoder_layers=2, n_decoder_layers=2, 
                                  max_src_len=seq_len+10, max_tgt_len=pred_len+10)
    
    t0 = time.time()
    train_loss, test_loss = train_eval_transformer(model, train_loader, test_loader, epochs=20, lr=1e-3)
    print(f"Training took {time.time() - t0:.1f}s")
    print(f"Final Train MSE: {train_loss[-1]:.4f} | Final Test MSE: {test_loss[-1]:.4f}")
    
    eval_tr = evaluate_portfolio(model, src_train, ret_train)
    eval_te = evaluate_portfolio(model, src_test, ret_test)
    
    print(f"Train Portfolio Return: {eval_tr['return_pct']:+.2f}% | BH: {eval_tr['bh_return_pct']:+.2f}% | Sharpe: {eval_tr['sharpe']:.3f}")
    print(f"Test Portfolio Return:  {eval_te['return_pct']:+.2f}% | BH: {eval_te['bh_return_pct']:+.2f}% | Sharpe: {eval_te['sharpe']:.3f}")
    
    gap_ret = eval_tr['return_pct'] - eval_te['return_pct']
    print(f"Return Gap: {gap_ret:+.2f}%")
    
    print("\n=== TEST 2: Permutation Test (Noise Learning) ===")
    # Shuffle targets
    tgt_train_shuffled = tgt_train[torch.randperm(tgt_train.size(0))]
    train_ds_shuffled = TensorDataset(src_train, tgt_train_shuffled)
    train_loader_shuffled = DataLoader(train_ds_shuffled, batch_size=64, shuffle=True)
    
    model_perm = TimeSeriesTransformer(input_dim=input_dim, d_model=64, n_heads=4, 
                                       n_encoder_layers=2, n_decoder_layers=2, 
                                       max_src_len=seq_len+10, max_tgt_len=pred_len+10)
    train_eval_transformer(model_perm, train_loader_shuffled, test_loader, epochs=20, lr=1e-3)
    
    eval_perm_tr = evaluate_portfolio(model_perm, src_train, ret_train)
    eval_perm_te = evaluate_portfolio(model_perm, src_test, ret_test)
    
    print(f"Permuted Train Return: {eval_perm_tr['return_pct']:+.2f}% | Sharpe: {eval_perm_tr['sharpe']:.3f}")
    print(f"Permuted Test Return:  {eval_perm_te['return_pct']:+.2f}% | Sharpe: {eval_perm_te['sharpe']:.3f}")
    
    print("\n=== TEST 3: Random Baseline ===")
    # Random weights
    rand_weights = np.random.uniform(0, 1, size=(len(ret_test), input_dim))
    rand_weights = rand_weights / rand_weights.sum(axis=1, keepdims=True)
    rand_ret = (rand_weights * ret_test).sum(axis=1)
    rand_cum = np.cumprod(1 + rand_ret)[-1] - 1
    rand_sharpe = np.mean(rand_ret) / (np.std(rand_ret) + 1e-8) * np.sqrt(252)
    print(f"Random Portfolio Return: {rand_cum*100:+.2f}% | Sharpe: {rand_sharpe:.3f}")
    
    results = {
        'data': {
            'n_tickers': len(valid_cols),
            'n_samples': len(returns),
            'assets': list(valid_cols),
        },
        'mse': {
            'train_final': train_loss[-1],
            'test_final': test_loss[-1],
        },
        'performance': {
            'train': eval_tr,
            'test': eval_te,
            'permuted_train': eval_perm_tr,
            'permuted_test': eval_perm_te,
            'random_test_return': rand_cum * 100,
            'random_test_sharpe': rand_sharpe
        }
    }
    
    # Save results to json for doc generation
    out_file = _root / 'docs/transformer/overfitting_results.json'
    with open(out_file, 'w') as f:
        # numpy types not serializable easily, so using default=str
        json.dump(results, f, default=str, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
