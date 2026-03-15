"""
Test suite for TabTransformer - validates architecture, forward pass, and training
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from tab_transformer import TabTransformer, MultiHeadAttention, TransformerBlock, FeedForward
from train_tab_transformer import TabTransformerTrainer

class TestTabTransformer:
    """Test suite for TabTransformer components"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.passed = 0
        self.failed = 0
    
    def log_test(self, test_name, passed, details=""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"       {details}")
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_embeddings(self):
        """Test embedding layer"""
        print("\n=== Testing Embeddings ===")
        try:
            from tab_transformer import Embeddings
            emb = Embeddings(input_dim=100, output_dim=32).to(self.device)
            x = torch.randint(0, 100, (16, 1)).to(self.device)
            out = emb(x[:, 0])
            self.log_test("Embeddings forward pass", out.shape == (16, 32), 
                         f"Output shape: {out.shape}")
        except Exception as e:
            self.log_test("Embeddings forward pass", False, str(e))
    
    def test_multihead_attention(self):
        """Test multi-head attention"""
        print("\n=== Testing MultiHeadAttention ===")
        try:
            dim = 256
            attn = MultiHeadAttention(dim=dim, heads=8, dim_head=64, dropout=0.1).to(self.device)
            x = torch.randn(16, 4, dim).to(self.device)  # (batch, seq_len, dim)
            out = attn(x)
            self.log_test("MHA forward pass", out.shape == x.shape, 
                         f"Input: {x.shape}, Output: {out.shape}")
        except Exception as e:
            self.log_test("MHA forward pass", False, str(e))
    
    def test_feedforward(self):
        """Test feed-forward network"""
        print("\n=== Testing FeedForward ===")
        try:
            dim = 256
            ff = FeedForward(dim=dim, hidden_dim=512, dropout=0.1).to(self.device)
            x = torch.randn(16, 4, dim).to(self.device)
            out = ff(x)
            self.log_test("FeedForward forward pass", out.shape == x.shape,
                         f"Input: {x.shape}, Output: {out.shape}")
        except Exception as e:
            self.log_test("FeedForward forward pass", False, str(e))
    
    def test_transformer_block(self):
        """Test transformer block"""
        print("\n=== Testing TransformerBlock ===")
        try:
            dim = 256
            tb = TransformerBlock(dim=dim, depth=3, heads=8, dim_head=64, mlp_dim=512, dropout=0.1).to(self.device)
            x = torch.randn(16, 4, dim).to(self.device)
            out = tb(x)
            self.log_test("TransformerBlock forward pass", out.shape == x.shape,
                         f"Input: {x.shape}, Output: {out.shape}")
        except Exception as e:
            self.log_test("TransformerBlock forward pass", False, str(e))
    
    def test_tab_transformer_forward(self):
        """Test TabTransformer forward pass"""
        print("\n=== Testing TabTransformer (WITH DEBUG) ===")
        try:
            categorical_dims = [50, 100, 20, 15]
            
            # Create model with debug enabled
            model = TabTransformer(
                num_numerical_features=10,
                num_categorical_features=4,
                categorical_dims=categorical_dims,
                embedding_dim=32,
                depth=4,
                heads=4,  # Must be <= num_categorical_features (4)
                dim_head=64,
                mlp_dim=256,
                num_classes=1,  # Single probability output
                dropout=0.1,
                debug=True  # Enable debug output
            ).to(self.device)
            
            batch_size = 32
            num_numerical = 10
            num_categorical = 4
            
            # Create dummy inputs - respect each categorical dimension
            numerical_features = torch.randn(batch_size, num_numerical).to(self.device)
            # Generate categorical features within valid range for each column
            categorical_features = torch.stack([
                torch.randint(0, dim, (batch_size,)) for dim in categorical_dims
            ], dim=1).to(self.device)
            
            # Forward pass
            probs = model(numerical_features, categorical_features)
            
            # Check output is probability (0-1) with correct shape
            is_valid_shape = probs.shape == (batch_size,)
            is_valid_range = (probs >= 0).all() and (probs <= 1).all()
            
            self.log_test("TabTransformer forward pass", is_valid_shape and is_valid_range,
                         f"Output shape: {probs.shape}, Range: [{probs.min():.3f}, {probs.max():.3f}]")
        except Exception as e:
            self.log_test("TabTransformer forward pass", False, str(e))
            import traceback
            traceback.print_exc()
    
    def test_tab_transformer_parameters(self):
        """Test that model parameters are trainable"""
        print("\n=== Testing Model Parameters ===")
        try:
            num_cat_features = 4
            model = TabTransformer(
                num_numerical_features=10,
                num_categorical_features=num_cat_features,
                categorical_dims=[50, 100, 20, 15],
                embedding_dim=32,
                depth=4,
                heads=min(4, num_cat_features),  # FIXED: heads must be <= num_categorical_features
                dim_head=64,
                mlp_dim=256,
                num_classes=2
            ).to(self.device)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.log_test("Model has trainable parameters", trainable_params > 0,
                         f"Total: {total_params:,}, Trainable: {trainable_params:,}")
        except Exception as e:
            self.log_test("Model parameters", False, str(e))
    
    def test_training_step(self):
        """Test a single training step"""
        print("\n=== Testing Training Step ===")
        try:
            categorical_dims = [50, 100, 20, 15]
            
            model = TabTransformer(
                num_numerical_features=10,
                num_categorical_features=4,
                categorical_dims=categorical_dims,
                embedding_dim=32,
                depth=4,
                heads=4,  # Fixed: must be <= num_categorical_features
                dim_head=64,
                mlp_dim=256,
                num_classes=1  # Single probability output
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()  # Binary cross entropy for probabilities
            
            batch_size = 32
            numerical_features = torch.randn(batch_size, 10).to(self.device)
            # Generate categorical features within valid range for each column
            categorical_features = torch.stack([
                torch.randint(0, dim, (batch_size,)) for dim in categorical_dims
            ], dim=1).to(self.device)
            labels = torch.rand(batch_size).to(self.device)  # Float labels for BCE
            
            # Forward pass
            probs = model(numerical_features, categorical_features)
            loss = criterion(probs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.log_test("Training step", loss.item() > 0,
                         f"Loss: {loss.item():.4f}")
        except Exception as e:
            self.log_test("Training step", False, str(e))
            import traceback
            traceback.print_exc()
    
    def test_synthetic_data_training(self):
        """Test end-to-end training with synthetic data"""
        print("\n=== Testing End-to-End Training ===")
        try:
            # Create synthetic data
            n_samples = 1000
            n_numerical = 10
            n_categorical = 4
            
            X_train_num = np.random.randn(n_samples, n_numerical).astype(np.float32)
            X_train_cat = np.random.randint(0, 50, (n_samples, n_categorical))
            y_train = np.random.randint(0, 2, n_samples).astype(np.float32)  # Float for BCE
            
            X_test_num = np.random.randn(n_samples // 4, n_numerical).astype(np.float32)
            X_test_cat = np.random.randint(0, 50, (n_samples // 4, n_categorical))
            y_test = np.random.randint(0, 2, n_samples // 4).astype(np.float32)
            
            # Create tensors
            X_train_num_t = torch.FloatTensor(X_train_num).to(self.device)
            X_train_cat_t = torch.LongTensor(X_train_cat).to(self.device)
            y_train_t = torch.FloatTensor(y_train).to(self.device)  # Float for BCE
            
            X_test_num_t = torch.FloatTensor(X_test_num).to(self.device)
            X_test_cat_t = torch.LongTensor(X_test_cat).to(self.device)
            y_test_t = torch.FloatTensor(y_test).to(self.device)
            
            # Create model
            model = TabTransformer(
                num_numerical_features=n_numerical,
                num_categorical_features=n_categorical,
                categorical_dims=[50] * n_categorical,
                embedding_dim=16,
                depth=2,
                heads=2,  # Fixed: must be <= n_categorical (4)
                dim_head=32,
                mlp_dim=128,
                num_classes=1  # Single probability output
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()  # Binary cross entropy
            
            # Train for 5 epochs
            initial_loss = None
            final_loss = None
            
            for epoch in range(5):
                model.train()
                probs = model(X_train_num_t, X_train_cat_t)
                loss = criterion(probs, y_train_t)
                
                if epoch == 0:
                    initial_loss = loss.item()
                if epoch == 4:
                    final_loss = loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                train_probs = model(X_train_num_t, X_train_cat_t)
                train_preds = (train_probs >= 0.5).float()
                train_acc = (train_preds == y_train_t).float().mean().item()
            
            loss_decreased = final_loss < initial_loss
            self.log_test("Synthetic data training", loss_decreased and train_acc > 0.4,
                         f"Initial loss: {initial_loss:.4f}, Final: {final_loss:.4f}, Acc: {train_acc:.4f}")
        except Exception as e:
            self.log_test("Synthetic data training", False, str(e))
            import traceback
            traceback.print_exc()
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("TABTRANSFORMER TEST SUITE")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        self.test_embeddings()
        self.test_multihead_attention()
        self.test_feedforward()
        self.test_transformer_block()
        self.test_tab_transformer_forward()
        self.test_tab_transformer_parameters()
        self.test_training_step()
        self.test_synthetic_data_training()
        
        print("\n" + "=" * 60)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 60)
        
        return self.failed == 0

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = TestTabTransformer(device=device)
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
