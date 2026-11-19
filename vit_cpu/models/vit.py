import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Chia ảnh thành các patch và ánh xạ tuyến tính lên không gian embedding."""

    def __init__(
        self,
        image_size=56,
        patch_size=4,
        in_channels=3,
        embed_dim=192
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size phải chia hết cho patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W] -> projection: [B, embed_dim, H/P, W/P]
        x = self.projection(x)
        # flatten không gian 2D: [B, embed_dim, num_patches]
        x = x.flatten(2)
        # chuyển về thứ tự token: [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Cơ chế Multi-Head Self-Attention của ViT."""

    def __init__(self, dim=192, num_heads=3):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # Tạo Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)

        # Context vector: Attention @ V
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, N, C)

        out = self.projection(out)
        return out


class MLP(nn.Module):
    """Feed-Forward Network (MLP) với hàm kích hoạt GELU."""

    def __init__(self, dim=192, hidden_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Khối Encoder của Transformer với cơ chế Pre-LayerNorm và Residual."""

    def __init__(self, dim=192, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)

    def forward(self, x):
        # Attention + Residual (Pre-Norm)
        x = x + self.attention(self.norm1(x))
        # MLP + Residual (Pre-Norm)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Mô hình Vision Transformer (ViT) tiêu chuẩn cho bài toán phân loại ảnh."""

    def __init__(
        self,
        image_size=56,
        patch_size=4,
        num_classes=9,
        embed_dim=192,
        num_heads=3,
        depth=6,
        mlp_ratio=4
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embedding.num_patches

        # CLS token học được
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding học được
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Danh sách các khối Encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Khởi tạo trọng số
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embedding(x)

        # 2. Ghép CLS token vào đầu chuỗi
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # 3. Thêm Positional embedding
        x = x + self.pos_embedding

        # 4. Qua các khối Transformer Encoder
        for block in self.encoder:
            x = block(x)

        # 5. Lấy biểu diễn của CLS token (token đầu tiên)
        cls_out = self.norm(x[:, 0])

        # 6. Classifier Head
        logits = self.head(cls_out)
        return logits
