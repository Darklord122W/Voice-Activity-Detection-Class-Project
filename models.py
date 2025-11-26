# ============================================================
# models.py
# ============================================================
# Assignment: Implement neural architectures for
# Voice Activity Detection (VAD) using PyTorch.
#
# Tasks:
#   1. Implement a Xavier initializer
#   2. Implement the LSTM-based VAD model
#   3. Implement the BiLSTM-based VAD model
#   4. Implement the CNN + LSTM hybrid VAD model
#   5. Complete the model builder function
# ============================================================

import torch
import torch.nn as nn


NUM_CLASSES = 2  # 0 = non-speech, 1 = speech


# ------------------------------------------------------------
# 1. Xavier Uniform Initialization Utility
# ------------------------------------------------------------
def init_weights_xavier(m):
    """
    Apply Xavier uniform initialization to all suitable layers.
    """
    # TODO: implement the Xavier initializer
    # -------- For Linear layers --------
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # -------- For Conv1d/Conv2d layers --------
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # -------- For LSTM layers --------
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Recommended for recurrent matrices
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


# ------------------------------------------------------------
# 2. LSTM-based VAD model
# ------------------------------------------------------------
class LSTMVad(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # TODO: define the LSTM-based architecture
        self.lstm=nn.LSTM(
            input_size= n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, NUM_CLASSES)
    def forward(self, x):
        """
        x: [B, T, F]
        returns: logits [B, T, 2]
        """
        # TODO: implement the forward pass
        out, _ = self.lstm(x)          # out: [B, T, H]
        logits = self.fc(out)          # logits: [B, T, 2]
        return logits


# ------------------------------------------------------------
# 3. Bi-directional LSTM-based VAD model
# ------------------------------------------------------------
class BiLSTMVad(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # TODO: define the BiLSTM-based architecture
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        # Hidden size is doubled in a BiLSTM
        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        """
        x: [B, T, F]
        returns: logits [B, T, 2]
        """
        # TODO: implement the forward pass
        out, _ = self.lstm(x)          # out: [B, T, 2H]
        logits = self.fc(out)          # [B, T, 2]
        return logits


# ------------------------------------------------------------
# 4. CNN + (Bi)LSTM hybrid VAD model
# ------------------------------------------------------------
class CNNLSTMVad(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        conv1_out: int = 16,
        conv2_out: int = 32,
        pool_kernel: int = 2,
    ):
        super().__init__()
        # TODO: define CNN feature extractor and related parameters
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel),

            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel),
        )
        # Will be created dynamically once we know feature dims
        self.lstm = None
        self.fc = None
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout = dropout        

    def _build_recurrent(self, features_per_step: int):
        """
        Build LSTM and FC layers once CNN feature dimensions are known.
        """
        # TODO: implement recurrent layer creation
        self.lstm = nn.LSTM(
            input_size=features_per_step,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0
        )

        lstm_out_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(lstm_out_size, NUM_CLASSES)

    def forward(self, x):
        """
        x: [B, T, F]
        returns: logits [B, T', 2]
        """
        # TODO: implement the forward pass
        # x: [B, T, F]
        B, T, F = x.shape

        # CNN expects: [B, 1, T, F]
        x = x.unsqueeze(1)

        cnn_out = self.cnn(x)              # [B, C, T', F']
        B, C, Tp, Fp = cnn_out.shape

        # Flatten for LSTM: [B, T', C * F']
        cnn_out = cnn_out.permute(0, 2, 1, 3).reshape(B, Tp, C * Fp)

        # Create LSTM when first known
        if self.lstm is None:
            self._build_recurrent(C * Fp)
            self.lstm = self.lstm.to(cnn_out.device)
            self.fc = self.fc.to(cnn_out.device)
            
        lstm_out, _ = self.lstm(cnn_out)   # [B, T', H or 2H]
        logits = self.fc(lstm_out)         # [B, T', 2]
        return logits

# ------------------------------------------------------------
# Model builder
# ------------------------------------------------------------
def build_model(model_type, n_features, hidden_size=128, num_layers=2, **kwargs):
    model_type = model_type.lower()

    if model_type == "lstm":
        return LSTMVad(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )

    elif model_type == "bilstm":
        return BiLSTMVad(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )

    elif model_type == "cnnlstm":
        return CNNLSTMVad(
            n_features=n_features,
            hidden_size=hidden_size,
            lstm_layers=num_layers,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from ['lstm', 'bilstm', 'cnnlstm']."
        )
