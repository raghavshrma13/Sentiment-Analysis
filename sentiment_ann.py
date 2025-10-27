import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def load_embeddings_bin(path):
    with open(path, 'rb') as f:
        n = struct.unpack('Q', f.read(8))[0]
        dim = struct.unpack('Q', f.read(8))[0]
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape(n, dim)

def load_labels(path, n_samples):
    labels = []
    with open(path, 'r') as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            sentiment = line.strip().split(',')[-1].replace('"', '')
            # Map sentiment values to 0-3 range
            if sentiment == '-1' or sentiment == 'Irrelevant':
                sentiment = 0  # Map irrelevant to 0
            elif sentiment == '0' or sentiment == 'Negative':
                sentiment = 1  # Map negative to 1
            elif sentiment == '1' or sentiment == 'Neutral':
                sentiment = 2  # Map neutral to 2
            else:  # '2' or 'Positive'
                sentiment = 3  # Map positive to 3
            labels.append(int(sentiment))
    return np.array(labels)

class SentimentANN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(SentimentANN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 4)  # 4 classes now (irrelevant, negative, neutral, positive)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.softmax(x)

def main():
    # Test different sizes
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        print(f"\n=== Training ANN with {size} samples ===")
        
        # Load embeddings
        emb_path = f'data/embeddings/train_onehot_{size}.bin'
        if not os.path.exists(emb_path):
            print(f"Error: {emb_path} not found!")
            continue
            
        X = load_embeddings_bin(emb_path)
        y = load_labels('data/raw/train_for_cpp.csv', size)
        
        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        
        # Initialize model
        input_size = X.shape[1]
        model = SentimentANN(input_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        epochs = 30
        batch_size = min(32, size)
        best_val_acc = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = len(X_train) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val).float().mean()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f'models/sentiment_ann_{size}.pth')
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}, Loss: {total_loss/num_batches:.4f}, Val Acc: {val_acc:.4f}')
        
        print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    main()