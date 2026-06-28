import math

import torch
from torch import Tensor, nn






class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLPSinusoidTimeEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        # Important: for activations, use ReLU or GELU. Need zero gradient zones for some reason.
        # ELU and SiLU don't work.
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, in_dim),
        )

        # self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                if i == len(self.net) - 1:  # Last layer
                    # Initialize to output slightly larger values
                    # This encourages non-zero rotations from the start
                    nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                    nn.init.uniform_(layer.bias, -0.1, 0.1)
                else:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                    # nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        # x_t: [B, N, D]
        # t: [B] or [1] (scalar time value)
        B = x_t.shape[0]
        orig_dim = x_t.shape[-1]

        x_t = x_t.flatten(1)
        if t.dim() == 0:
            t = t.view(1).expand(B)
        if t.dim() == 2:
            t = t.squeeze(-1)

        t_emb = self.time_embed(t)
        input_x = torch.cat((x_t, t_emb), dim=-1)
        out = self.net(input_x)
        if out.shape[-1] == orig_dim * orig_dim:
            out = out.view(B, orig_dim, orig_dim)

        return out



class TransformerTimeEmbedding(nn.Module):
    """
    Transformer network with time embedding for flow matching on point clouds.
    
    Architecture:
    - Sinusoidal time embedding
    - Point-wise position encoding + time encoding
    - Multi-head self-attention (Transformer encoder)
    - Mean pooling over points
    - Output MLP for transformation prediction
    
    Args:
        in_dim: Input dimension per point (default: 3 for 3D points)
        out_dim: Output dimension (3 for Lie algebra, 9 for 3x3 matrix)
        hidden_dim: Hidden dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        time_embed_dim: Time embedding dimension (default: 128)
    """
    
    def __init__(
        self,
        in_dim=3,
        out_dim=3,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        time_embed_dim=128,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        
        # Time embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_embed_dim),
        )
        
        # Input projection: point positions + time -> hidden_dim
        self.input_projection = nn.Linear(in_dim + time_embed_dim, hidden_dim)
        
        # Learnable positional encoding for token positions
        # Support up to 2048 points
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, hidden_dim) * 0.02)
        
        # Transformer encoder with pre-norm for better stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass with time embedding.
        
        Args:
            x_t: Point cloud [B, N, D] where B is batch size, N is number of points, D is point dimension
            t: Time [B] or scalar
        
        Returns:
            Transformation: [B, 3, 3] matrix or [B, 3] Lie algebra element
        """
        B, N, D = x_t.shape
        
        # Expand time to match batch size
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(B)
        
        # Time embedding: [B] -> [B, time_embed_dim]
        t_emb = self.time_embed(t)
        
        # Expand time embedding to all points: [B, time_embed_dim] -> [B, N, time_embed_dim]
        t_emb_expanded = t_emb.unsqueeze(1).expand(B, N, self.time_embed_dim)
        
        # Concatenate point positions with time embedding: [B, N, D + time_embed_dim]
        x_with_time = torch.cat([x_t, t_emb_expanded], dim=-1)
        
        # Project to hidden dimension: [B, N, hidden_dim]
        x = self.input_projection(x_with_time)
        
        # Add learnable positional encoding
        x = x + self.pos_encoding[:, :N, :]
        
        # Apply Transformer encoder: [B, N, hidden_dim] -> [B, N, hidden_dim]
        x = self.transformer(x)
        
        # Mean pooling over points: [B, N, hidden_dim] -> [B, hidden_dim]
        pooled = x.mean(dim=1)
        
        # Output MLP: [B, hidden_dim] -> [B, out_dim]
        output = self.output_mlp(pooled)
        
        # Reshape output based on out_dim
        if self.out_dim == 3:
            # Lie algebra element (3-vector)
            output = output.view(B, -1)
        elif self.out_dim == 9:
            # Transformation matrix (3x3)
            output = output.view(B, 3, 3)
        elif self.out_dim == 12:
            # 3x4 transformation matrix
            output = output.view(B, 3, 4)
        elif self.out_dim == 16:
            # 4x4 transformation matrix
            output = output.view(B, 4, 4)
        else:
            # Keep as is
            output = output.view(B, -1)
        
        return output




