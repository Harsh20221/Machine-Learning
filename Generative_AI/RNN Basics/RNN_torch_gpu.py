import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
    print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")

# Enable cuDNN auto-tuner for faster training
torch.backends.cudnn.benchmark = True

# Define the RNN model
class IMDBRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        # text shape: (batch_size, sequence_length)
        embedded = self.embedding(text)  # (batch_size, sequence_length, embedding_dim)
        output, hidden = self.rnn(embedded)
        # Use the output of the last time step
        out = self.fc(output[:, -1, :])
        out = self.sigmoid(out)
        return out

# Hyperparameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 2
BATCH_SIZE = 256  # Increased for better GPU utilization
EPOCHS = 10
MAX_LENGTH = 200
LEARNING_RATE = 0.001

# Create model instance and move to GPU
model = IMDBRNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS).to(device)
print("\nModel Architecture:")
print(model)

# Generate some dummy data for testing GPU functionality
def generate_dummy_data(num_samples):
    # Generate random sequences
    sequences = torch.randint(0, VOCAB_SIZE, (num_samples, MAX_LENGTH)).to(device)
    # Generate random labels (0 or 1)
    labels = torch.randint(0, 2, (num_samples,)).float().to(device)
    return sequences, labels

# Create data loaders with dummy data
train_sequences, train_labels = generate_dummy_data(50000)  # 50,000 training samples
test_sequences, test_labels = generate_dummy_data(10000)    # 10,000 test samples

train_data = torch.utils.data.TensorDataset(train_sequences, train_labels)
test_data = torch.utils.data.TensorDataset(test_sequences, test_labels)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\nStarting training...")
best_accuracy = 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        total_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch+1}/{EPOCHS}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
            
            # Print GPU memory usage
            if torch.cuda.is_available():
                print(f"GPU Memory Usage:")
                print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
                print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            predicted = (outputs.squeeze() > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    accuracy = 100. * val_correct / val_total
    print(f'\nEpoch: {epoch+1}/{EPOCHS}, '
          f'Training Loss: {total_loss/len(train_loader):.4f}, '
          f'Validation Accuracy: {accuracy:.2f}%')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

training_time = time.time() - start_time
print(f'\nTraining completed in {training_time/60:.2f} minutes')
print(f'Best validation accuracy: {best_accuracy:.2f}%')
