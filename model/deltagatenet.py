import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalDelta(nn.Module):
    """
    Bidirectional first-order temporal differencing.
    Separates positive and negative changes.

    Input : (B, C, T)
    Output: (B, 2*C, T)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        delta = x[:, :, 1:] - x[:, :, :-1]
        delta = F.pad(delta, (1, 0))

        delta_pos = F.relu(delta)
        delta_neg = F.relu(-delta)

        return torch.cat([delta_pos, delta_neg], dim=1)


BirectionalDelta = BidirectionalDelta


class GatedTemporalConv(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        hidden_dims = 16
        num_layers = 2
        dropout = 0.5
        kernel_size = 7

        self.input_channels = input_channels
        self.hidden_dims = hidden_dims

        self.input_proj = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels * hidden_dims,
            kernel_size=1,
            groups=1,
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_channels * hidden_dims,
                        out_channels=input_channels * hidden_dims,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=1,
                    ),
                    nn.BatchNorm1d(input_channels * hidden_dims),
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=input_channels * hidden_dims,
                        out_channels=input_channels * hidden_dims,
                        kernel_size=1,
                    ),
                    nn.Dropout(dropout),
                )
            )

        self.norm = nn.LayerNorm(hidden_dims)

    def forward(self, x):
        """
        x: (B, C, T)
        return: (B, C, hidden_dims)
        """
        B, C, T = x.shape

        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer(x)

        x = x.view(B, C, self.hidden_dims, T)
        x = x.mean(dim=-1)
        x = self.norm(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()

        hidden_dims = 16
        dropout_rate = 0.5

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims, num_classes),
        )

    def forward(self, input):
        return self.mlp(input)


class DeltaGateNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(DeltaGateNet, self).__init__()

        hidden_dims = 16

        self.temporal_diff = BidirectionalDelta()
        self.conv = GatedTemporalConv(input_channels=2 * num_channels)
        self.mlp = MLP(
            input_dim=2 * num_channels * hidden_dims,
            num_classes=num_classes,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, eeg):
        batch_size = eeg.size(0)
        eeg = self.temporal_diff(eeg)
        eeg_features = self.conv(eeg)
        eeg_flat = eeg_features.view(batch_size, -1)
        return self.mlp(eeg_flat)
