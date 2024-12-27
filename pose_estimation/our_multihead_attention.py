import torch
import math

def scaled_attention_product(q, k, mask=None):
    d_k = q.size(-1)  # Head dimension
    # Compute scaled dot-product attention logits
    attn_logits = torch.matmul(q, k.transpose(-2, -1))  # [B, H, Lq, Lk]
    attn_logits = attn_logits / math.sqrt(d_k)

    if mask is not None:
        # Expand mask if necessary and apply it
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

    # For numerical stability, subtract the max from the logits before applying softmax
    attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]

    # Apply softmax to get the attention weights
    attention = torch.nn.functional.softmax(attn_logits, dim=-1)  # [B, H, Lq, Lk]
    return attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, ray_fea_size, img_fea_size, embed_dim, num_heads=1):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for projecting inputs to queries and keys
        self.q_proj = torch.nn.Linear(img_fea_size, embed_dim)
        self.k_proj = torch.nn.Linear(ray_fea_size, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters using Xavier uniform initialization
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)

    def forward(self, img_features, ray_features, mask=None):
        # Handle cases where img_features and ray_features are 2D tensors
        if img_features.dim() == 2:
            img_features = img_features.unsqueeze(0)  # Shape: [1, seq_len_img, img_fea_size]
        if ray_features.dim() == 2:
            ray_features = ray_features.unsqueeze(0)  # Shape: [1, seq_len_ray, ray_fea_size]

        batch_size_img, seq_len_img, _ = img_features.size()
        batch_size_ray, seq_len_ray, _ = ray_features.size()

        # Ensure batch sizes match or handle accordingly
        if batch_size_img != batch_size_ray:
            raise ValueError("Batch sizes of img_features and ray_features must match.")

        # Project the inputs
        q = self.q_proj(img_features)     # [batch_size, seq_len_img, embed_dim]
        k = self.k_proj(ray_features)     # [batch_size, seq_len_ray, embed_dim]

        # Reshape and transpose to get [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size_img, seq_len_img, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_img, head_dim]
        k = k.view(batch_size_ray, seq_len_ray, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_ray, head_dim]

        # Compute attention weights
        attention = scaled_attention_product(q, k, mask=mask)  # [batch_size, num_heads, seq_len_img, seq_len_ray]

        # Remove batch dimension if originally absent
        if attention.size(0) == 1:
            attention = attention.squeeze(0)  # [num_heads, seq_len_img, seq_len_ray]

        return attention
