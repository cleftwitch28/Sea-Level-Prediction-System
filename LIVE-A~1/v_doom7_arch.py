"""V_DOOM7 architecture extracted from Colab notebook 1_F754g5J8kjtGJ-BvFkSDoiuy4I2TixH.

Full inheritance chain: SeaLevelModel_V18 (V_DOOM7) inherits from V14 -> V13 -> V11 -> V9 -> V8 -> V7 -> V6 -> V5 -> V3 -> V2.
Human branch: HumanFactorsNet_V8 (four pathways: anthro, coastal, ocean, s1).
Fusion: CrossAttentionFusion_V2 (per-branch LayerNorm before cross-attention).

Constants:
    EMBED_DIM = 128
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EMBED_DIM = 128
SEQ_LEN = 365
SPATIAL_C = 5
HUMAN_C = 4
TEMP_F = 8
SHORT_C = 1
N_REGIONS = 10
# Extra constants used inside class bodies (extracted from Colab context)
TEMP_T = SEQ_LEN         # temporal sequence length
HUMAN_F = 17             # human features per timestep
HUMAN_IMG_C = 3          # human-branch image input channels (RGB)


class SEBlock(nn.Module):

    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels // r), nn.ReLU(), nn.Linear(channels // r, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class SE3DBlock(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps. Learns which channels matter."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):
        (b, c) = x.shape[:2]
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1, 1)
        return x * s

class AuxHead(nn.Module):
    """Per-branch auxiliary regression head -- prevents branch collapse."""

    def __init__(self, emb=EMBED_DIM, hidden=64):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.head(z)

class PhysicsHead(nn.Module):
    """Single-scalar predictor for one physics component."""

    def __init__(self, emb=EMBED_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: out = gamma * x + beta, with gamma, beta
    predicted from a conditioning vector. For 3D conv outputs (B, C, T, H, W)."""

    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, n_channels)
        self.beta = nn.Linear(cond_dim, n_channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        g = self.gamma(cond).view(*cond.shape[:1], -1, 1, 1, 1)
        b = self.beta(cond).view(*cond.shape[:1], -1, 1, 1, 1)
        return g * x + b

class CrossAttentionFusion(nn.Module):
    """Branch tokens + CLS token -> transformer -> heteroscedastic (mu, log_var)."""

    def __init__(self, emb=EMBED_DIM, n_heads=4, n_layers=2):
        super().__init__()
        self.type_emb = nn.Parameter(torch.randn(4, emb) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, emb) * 0.02)
        layer = nn.TransformerEncoderLayer(emb, n_heads, dim_feedforward=4 * emb, dropout=0.15, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mu_head = nn.Sequential(nn.Linear(emb, emb // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(emb // 2, 1))
        self.var_head = nn.Sequential(nn.Linear(emb, emb // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(emb // 2, 1))

    def forward(self, zs, zh, zt, zr):
        B = zs.size(0)
        branches = torch.stack([zs, zh, zt, zr], dim=1) + self.type_emb.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, branches], dim=1)
        attended = self.encoder(tokens)
        return (self.mu_head(attended[:, 0]), self.var_head(attended[:, 0]))

class CrossAttentionFusion_V2(CrossAttentionFusion):
    """V1 fusion + per-branch LayerNorm to neutralize magnitude differences across branches.
    Each branch token is normalized BEFORE the type-embedding add and cross-attention,
    so a numerically large branch (e.g. temporal) cannot dominate by norm alone."""

    def __init__(self, emb=EMBED_DIM, n_heads=4, n_layers=2):
        super().__init__(emb=emb, n_heads=n_heads, n_layers=n_layers)
        self.token_norms = nn.ModuleList([nn.LayerNorm(emb) for _ in range(4)])

    def forward(self, zs, zh, zt, zr):
        zs = self.token_norms[0](zs)
        zh = self.token_norms[1](zh)
        zt = self.token_norms[2](zt)
        zr = self.token_norms[3](zr)
        return super().forward(zs, zh, zt, zr)

class SpatialST3DCNN(nn.Module):
    """3D-CNN over (T, C, H, W) spatial-temporal stack."""

    def __init__(self, in_ch=5, embed=EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv3d(in_ch, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(16), nn.ReLU(), nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(32), nn.ReLU(), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.ReLU(), nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1, stride=(1, 2, 2)), nn.BatchNorm3d(128), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(128, embed), nn.ReLU())
        self.aux = AuxHead(embed)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encoder(x)
        return (z, self.aux(z))

class TemporalNet_V1(nn.Module):
    """Transformer + GRU over long temporal ocean series."""

    def __init__(self, in_dim=TEMP_F, emb=EMBED_DIM, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, TEMP_T, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.15, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.gru = nn.GRU(d_model, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, emb), nn.ReLU())
        self.aux = AuxHead(emb)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)
        (_, h) = self.gru(x)
        z = self.head(h[-1])
        return (z, self.aux(z))

class ShortTermNet_V1(nn.Module):
    """CNN over event frames + LSTM over time."""

    def __init__(self, in_c=SHORT_C, emb=EMBED_DIM):
        super().__init__()
        self.frame_cnn = nn.Sequential(nn.Conv2d(in_c, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), SEBlock(64), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(128, emb), nn.ReLU())
        self.aux = AuxHead(emb)

    def forward(self, x):
        (B, T) = x.shape[:2]
        f = self.frame_cnn(x.view(B * T, *x.shape[2:])).view(B, T, 128)
        (_, (h, _)) = self.lstm(f)
        z = self.head(h[-1])
        return (z, self.aux(z))

class HumanFactorsNet_V1(nn.Module):
    """LSTM over coastal frames + GRU over GHG/climate features."""

    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.frame_cnn = nn.Sequential(nn.Conv2d(HUMAN_IMG_C, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), SEBlock(64), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.frame_lstm = nn.LSTM(128, 128, batch_first=True)
        self.emis_gru = nn.GRU(HUMAN_F, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(128 + 64, emb), nn.ReLU())
        self.aux = AuxHead(emb)

    def forward(self, emissions, coastal):
        (B, T) = coastal.shape[:2]
        f = self.frame_cnn(coastal.view(B * T, *coastal.shape[2:])).view(B, T, 128)
        (_, (h_lstm, _)) = self.frame_lstm(f)
        (_, h_gru) = self.emis_gru(emissions)
        z = self.head(torch.cat([h_lstm[-1], h_gru[-1]], dim=1))
        return (z, self.aux(z))

class HumanFactorsNet_V3(nn.Module):
    """GRACE 2D imagery as primary input; 17 numerical features as FiLM conditioning.
    Architecture:
      grace_2d (B, 12, 1, 64, 64) -> 3D-CNN -> feature maps (B, 64, T', H', W')
      emissions (B, 24, 17)       -> MLP    -> FiLM cond vector (B, 64)
      FiLM modulation             -> pool   -> 128-dim embedding
    Coastal imagery (which we showed to be uninformative) is ignored."""

    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.grace_cnn = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(16), nn.GELU(), nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU())
        self.num_encoder = nn.Sequential(nn.Flatten(), nn.Linear(24 * HUMAN_F, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.film = FiLM(cond_dim=64, n_channels=64)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(nn.Linear(64, emb), nn.GELU(), nn.LayerNorm(emb))
        self.aux = AuxHead(emb)

    def forward(self, emissions, coastal, grace_2d):
        x = grace_2d.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.grace_cnn(x)
        cond = self.num_encoder(emissions)
        modulated = self.film(feat, cond)
        z = self.pool(modulated).flatten(1)
        z = self.proj(z)
        return (z, self.aux(z))

class HumanFactorsNet_V4(nn.Module):
    """Same FiLM architecture as V3, but accepts 5-channel imagery (GRACE+NTL+AOD+Pop+Industrial)."""

    def __init__(self, emb=EMBED_DIM, in_channels=5):
        super().__init__()
        self.anthro_cnn = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU())
        self.num_encoder = nn.Sequential(nn.Flatten(), nn.Linear(24 * HUMAN_F, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.film = FiLM(cond_dim=64, n_channels=64)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(nn.Linear(64, emb), nn.GELU(), nn.LayerNorm(emb))
        self.aux = AuxHead(emb)

    def forward(self, emissions, coastal, anthro_imagery):
        x = anthro_imagery.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.anthro_cnn(x)
        cond = self.num_encoder(emissions)
        modulated = self.film(feat, cond)
        z = self.pool(modulated).flatten(1)
        z = self.proj(z)
        return (z, self.aux(z))

class HumanFactorsNet_V5(HumanFactorsNet_V4):

    def __init__(self, emb=EMBED_DIM):
        super().__init__(emb=emb, in_channels=4)

class HumanFactorsNet_V6(nn.Module):
    """The Synthesizer: best-of-all-lessons human branch.
    Inputs:
      emissions:    (B, 24, HUMAN_F)   numerical features
      coastal:      (B, 3, 64, 64)     static coastal imagery
      anthro_5ch:   (B, 12, 5, 64, 64) GRACE+NTL+Pop+Industrial+SST_TREND
    Outputs:
      z:            (B, emb)           branch embedding for fusion
      y_aux:        (B, 1)             aux head prediction of y
      residual_pred: (B, 1)            direct prediction of (y - V3.0_prediction)
    """

    def __init__(self, emb=EMBED_DIM, in_channels=5):
        super().__init__()
        self.gru = nn.GRU(HUMAN_F, 64, batch_first=True, num_layers=2, dropout=0.1)
        self.long_proj = nn.Sequential(nn.Linear(HUMAN_F, 32), nn.GELU(), nn.LayerNorm(32))
        self.anthro_cnn = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), SE3DBlock(64), nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(96), nn.GELU())
        self.film = FiLM(cond_dim=64 + 32, n_channels=96)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.token_dim = 24
        self.n_tokens = 96 // self.token_dim
        self.attn = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(self.token_dim)
        self.proj = nn.Sequential(nn.Linear(96, emb), nn.GELU(), nn.LayerNorm(emb))
        self.aux = AuxHead(emb)
        self.residual_head = nn.Sequential(nn.Linear(emb, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, emissions, coastal, anthro_5ch):
        (_, h) = self.gru(emissions)
        h = h[-1]
        long_feat = self.long_proj(emissions.mean(dim=1))
        cond = torch.cat([h, long_feat], dim=1)
        x = anthro_5ch.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.anthro_cnn(x)
        modulated = self.film(feat, cond)
        pooled = self.pool(modulated).flatten(1)
        tokens = pooled.view(pooled.size(0), self.n_tokens, self.token_dim)
        (attended, _) = self.attn(tokens, tokens, tokens)
        attended = self.attn_norm(attended + tokens)
        z_pre = attended.flatten(1)
        z = self.proj(z_pre)
        residual = self.residual_head(z)
        return (z, self.aux(z), residual)

class HumanFactorsNet_V6_4ch(HumanFactorsNet_V6):

    def __init__(self, emb=EMBED_DIM):
        super().__init__(emb=emb, in_channels=4)

class HumanFactorsNet_V7(nn.Module):

    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.anthro_cnn = nn.Sequential(nn.Conv3d(4, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), SE3DBlock(64))
        self.coastal_cnn = nn.Sequential(nn.Conv3d(3, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), SE3DBlock(64))
        self.ocean_cnn = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(16), nn.GELU(), nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32))
        self.combined_se = SE3DBlock(160)
        self.gru = nn.GRU(HUMAN_F, 64, batch_first=True, num_layers=2, dropout=0.1)
        self.long_proj = nn.Sequential(nn.Linear(HUMAN_F, 32), nn.GELU(), nn.LayerNorm(32))
        self.film = FiLM(cond_dim=64 + 32, n_channels=160)
        self.pool = nn.AdaptiveAvgPool3d(1)
        (self.token_dim, self.n_tokens) = (40, 4)
        self.attn = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(self.token_dim)
        self.proj = nn.Sequential(nn.Linear(160, emb), nn.GELU(), nn.LayerNorm(emb))
        self.aux = AuxHead(emb)
        self.residual_head = nn.Sequential(nn.Linear(emb, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, emissions, coastal_static, anthro_4ch, coastal_timelapse, ocean_chl):
        a = self.anthro_cnn(anthro_4ch.permute(0, 2, 1, 3, 4).contiguous())
        c = self.coastal_cnn(coastal_timelapse.permute(0, 2, 1, 3, 4).contiguous())
        o = self.ocean_cnn(ocean_chl.permute(0, 2, 1, 3, 4).contiguous())
        T_min = min(a.shape[2], c.shape[2], o.shape[2])
        H_min = min(a.shape[3], c.shape[3], o.shape[3])
        W_min = min(a.shape[4], c.shape[4], o.shape[4])
        a = a[:, :, :T_min, :H_min, :W_min]
        c = c[:, :, :T_min, :H_min, :W_min]
        o = o[:, :, :T_min, :H_min, :W_min]
        combined = torch.cat([a, c, o], dim=1)
        combined = self.combined_se(combined)
        (_, h) = self.gru(emissions)
        h = h[-1]
        long_feat = self.long_proj(emissions.mean(dim=1))
        cond = torch.cat([h, long_feat], dim=1)
        modulated = self.film(combined, cond)
        pooled = self.pool(modulated).flatten(1)
        tokens = pooled.view(pooled.size(0), self.n_tokens, self.token_dim)
        (attended, _) = self.attn(tokens, tokens, tokens)
        attended = self.attn_norm(attended + tokens)
        z_pre = attended.flatten(1)
        z = self.proj(z_pre)
        return (z, self.aux(z), self.residual_head(z))

class HumanFactorsNet_V7b(nn.Module):
    """2-pathway: anthropogenic 2ch + coastal_TL 3ch + numerical FiLM cond."""

    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.anthro_cnn = nn.Sequential(nn.Conv3d(2, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), SE3DBlock(64))
        self.coastal_cnn = nn.Sequential(nn.Conv3d(3, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), SE3DBlock(64))
        self.combined_se = SE3DBlock(128)
        self.gru = nn.GRU(HUMAN_F, 64, batch_first=True, num_layers=2, dropout=0.1)
        self.long_proj = nn.Sequential(nn.Linear(HUMAN_F, 32), nn.GELU(), nn.LayerNorm(32))
        self.film = FiLM(cond_dim=64 + 32, n_channels=128)
        self.pool = nn.AdaptiveAvgPool3d(1)
        (self.token_dim, self.n_tokens) = (32, 4)
        self.attn = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(self.token_dim)
        self.proj = nn.Sequential(nn.Linear(128, emb), nn.GELU(), nn.LayerNorm(emb))
        self.aux = AuxHead(emb)
        self.residual_head = nn.Sequential(nn.Linear(emb, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, emissions, coastal_static, anthro_2ch, coastal_timelapse):
        a = self.anthro_cnn(anthro_2ch.permute(0, 2, 1, 3, 4).contiguous())
        c = self.coastal_cnn(coastal_timelapse.permute(0, 2, 1, 3, 4).contiguous())
        (T_min, H_min, W_min) = (min(a.shape[2], c.shape[2]), min(a.shape[3], c.shape[3]), min(a.shape[4], c.shape[4]))
        a = a[:, :, :T_min, :H_min, :W_min]
        c = c[:, :, :T_min, :H_min, :W_min]
        combined = torch.cat([a, c], dim=1)
        combined = self.combined_se(combined)
        (_, h) = self.gru(emissions)
        h = h[-1]
        long_feat = self.long_proj(emissions.mean(dim=1))
        cond = torch.cat([h, long_feat], dim=1)
        modulated = self.film(combined, cond)
        pooled = self.pool(modulated).flatten(1)
        tokens = pooled.view(pooled.size(0), self.n_tokens, self.token_dim)
        (attended, _) = self.attn(tokens, tokens, tokens)
        attended = self.attn_norm(attended + tokens)
        z_pre = attended.flatten(1)
        z = self.proj(z_pre)
        return (z, self.aux(z), self.residual_head(z))

class HumanFactorsNet_V8(nn.Module):

    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.anthro_cnn = nn.Sequential(nn.Conv3d(4, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), SE3DBlock(64))
        self.coastal_cnn = nn.Sequential(nn.Conv3d(3, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32), nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.GELU(), SE3DBlock(64))
        self.ocean_cnn = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(16), nn.GELU(), nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32))
        self.s1_cnn = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)), nn.BatchNorm3d(16), nn.GELU(), nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), SE3DBlock(32))
        self.combined_se = SE3DBlock(192)
        self.gru = nn.GRU(HUMAN_F, 64, batch_first=True, num_layers=2, dropout=0.1)
        self.long_proj = nn.Sequential(nn.Linear(HUMAN_F, 32), nn.GELU(), nn.LayerNorm(32))
        self.film = FiLM(cond_dim=64 + 32, n_channels=192)
        self.pool = nn.AdaptiveAvgPool3d(1)
        (self.token_dim, self.n_tokens) = (32, 6)
        self.attn = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(self.token_dim)
        self.proj = nn.Sequential(nn.Linear(192, emb), nn.GELU(), nn.LayerNorm(emb))
        self.aux = AuxHead(emb)
        self.residual_head = nn.Sequential(nn.Linear(emb, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, emissions, coastal_static, anthro_4ch, coastal_timelapse, ocean_chl, s1_sar):
        a = self.anthro_cnn(anthro_4ch.permute(0, 2, 1, 3, 4).contiguous())
        c = self.coastal_cnn(coastal_timelapse.permute(0, 2, 1, 3, 4).contiguous())
        o = self.ocean_cnn(ocean_chl.permute(0, 2, 1, 3, 4).contiguous())
        s = self.s1_cnn(s1_sar.permute(0, 2, 1, 3, 4).contiguous())
        T_min = min(a.shape[2], c.shape[2], o.shape[2], s.shape[2])
        H_min = min(a.shape[3], c.shape[3], o.shape[3], s.shape[3])
        W_min = min(a.shape[4], c.shape[4], o.shape[4], s.shape[4])
        a = a[:, :, :T_min, :H_min, :W_min]
        c = c[:, :, :T_min, :H_min, :W_min]
        o = o[:, :, :T_min, :H_min, :W_min]
        s = s[:, :, :T_min, :H_min, :W_min]
        combined = torch.cat([a, c, o, s], dim=1)
        combined = self.combined_se(combined)
        (_, h) = self.gru(emissions)
        h = h[-1]
        long_feat = self.long_proj(emissions.mean(dim=1))
        cond = torch.cat([h, long_feat], dim=1)
        modulated = self.film(combined, cond)
        pooled = self.pool(modulated).flatten(1)
        tokens = pooled.view(pooled.size(0), self.n_tokens, self.token_dim)
        (attended, _) = self.attn(tokens, tokens, tokens)
        attended = self.attn_norm(attended + tokens)
        z_pre = attended.flatten(1)
        z = self.proj(z_pre)
        return (z, self.aux(z), self.residual_head(z))

class GunpointFusion(CrossAttentionFusion):
    """V2 fusion (LayerNorm tokens) + forced additive human contribution.

    Final mu = mu_fusion_CLS + gate * mu_human_direct
        gate = gate_floor + sigmoid(raw_gate)   →   gate ∈ [floor, floor+1]

    The model can lower `gate` toward `gate_floor` if human is unhelpful,
    but cannot zero it out. Forced minimum contribution = gate_floor (e.g. 0.20).
    Uses the RAW human embedding (pre-LayerNorm) for the direct prediction so
    the human branch's scale information is preserved."""

    def __init__(self, emb=EMBED_DIM, n_heads=4, n_layers=2, gate_floor=0.2):
        super().__init__(emb=emb, n_heads=n_heads, n_layers=n_layers)
        self.token_norms = nn.ModuleList([nn.LayerNorm(emb) for _ in range(4)])
        self.human_direct = nn.Sequential(nn.Linear(emb, emb // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(emb // 2, 1))
        self.raw_gate = nn.Parameter(torch.tensor(0.0))
        self.gate_floor = gate_floor

    def forward(self, zs, zh, zt, zr):
        zs_n = self.token_norms[0](zs)
        zh_n = self.token_norms[1](zh)
        zt_n = self.token_norms[2](zt)
        zr_n = self.token_norms[3](zr)
        B = zs_n.size(0)
        branches = torch.stack([zs_n, zh_n, zt_n, zr_n], dim=1) + self.type_emb.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, branches], dim=1)
        attended = self.encoder(tokens)
        cls_out = attended[:, 0]
        mu_base = self.mu_head(cls_out)
        log_var = self.var_head(cls_out)
        mu_human_direct = self.human_direct(zh)
        gate = self.gate_floor + torch.sigmoid(self.raw_gate)
        mu_final = mu_base + gate * mu_human_direct
        return (mu_final, log_var)

class StrictGunpointFusion(CrossAttentionFusion):
    """Human token is REMOVED from cross-attention CLS routing.
    The ONLY way human information enters mu_final is via the gated direct head.
    Fusion CANNOT compensate — it never sees the human embedding in its routing."""

    def __init__(self, emb=EMBED_DIM, n_heads=4, n_layers=2, gate_floor=0.3):
        super().__init__(emb=emb, n_heads=n_heads, n_layers=n_layers)
        self.type_emb_3 = nn.Parameter(torch.randn(3, emb) * 0.02)
        self.token_norms = nn.ModuleList([nn.LayerNorm(emb) for _ in range(4)])
        self.human_direct = nn.Sequential(nn.Linear(emb, emb // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(emb // 2, 1))
        self.raw_gate = nn.Parameter(torch.tensor(1.0))
        self.gate_floor = gate_floor

    def forward(self, zs, zh, zt, zr):
        zs_n = self.token_norms[0](zs)
        zh_n = self.token_norms[1](zh)
        zt_n = self.token_norms[2](zt)
        zr_n = self.token_norms[3](zr)
        B = zs_n.size(0)
        branches = torch.stack([zs_n, zt_n, zr_n], dim=1) + self.type_emb_3.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, branches], dim=1)
        attended = self.encoder(tokens)
        cls_out = attended[:, 0]
        mu_base = self.mu_head(cls_out)
        log_var = self.var_head(cls_out)
        mu_human_direct = self.human_direct(zh_n)
        gate = self.gate_floor + torch.sigmoid(self.raw_gate)
        mu_final = mu_base + gate * mu_human_direct
        return (mu_final, log_var)

class HumanOnlyResidual(nn.Module):

    def __init__(self):
        super().__init__()
        self.anthro_cnn = nn.Sequential(nn.Conv3d(4, 16, kernel_size=(3, 3, 3), padding=1, stride=(1, 2, 2)), nn.BatchNorm3d(16), nn.GELU(), nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 2)), nn.BatchNorm3d(32), nn.GELU(), nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.em_mlp = nn.Sequential(nn.Flatten(), nn.Linear(24 * HUMAN_F, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 32))
        self.head = nn.Sequential(nn.Linear(32 + 32, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, anthro_4ch, emissions):
        x = anthro_4ch.permute(0, 2, 1, 3, 4).contiguous()
        z_img = self.anthro_cnn(x)
        z_em = self.em_mlp(emissions)
        return self.head(torch.cat([z_img, z_em], dim=1)).squeeze(-1)

class SeaLevelModel_V2(nn.Module):

    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.spatial = SpatialST3DCNN(embed=emb)
        self.human = HumanFactorsNet_V1(emb=emb)
        self.temporal = TemporalNet_V1(emb=emb)
        self.short = ShortTermNet_V1(emb=emb)
        self.fuse = CrossAttentionFusion(emb=emb)

    def forward(self, spatial, emissions, coastal, temporal, short_term):
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        return {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1)}

class SeaLevelModel_V3(SeaLevelModel_V2):
    """V2 model + V2 fusion (LayerNorm normalization at fusion entry).
    All four branch encoders are unchanged — only the fusion layer is replaced."""

    def __init__(self, emb=EMBED_DIM):
        super().__init__(emb=emb)
        self.fuse = CrossAttentionFusion_V2(emb=emb)

    def forward_with_embeddings(self, spatial, emissions, coastal, temporal, short_term):
        """Same as forward(), but also returns the raw branch embeddings for diversity loss."""
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        return ({'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1)}, {'spatial': zs, 'human': zh, 'temporal': zt, 'short': zr})

class SeaLevelModel_V5(SeaLevelModel_V3):
    """V3 (LayerNorm fusion) with the human branch replaced by HumanFactorsNet_V3."""

    def __init__(self, emb=EMBED_DIM):
        super().__init__(emb=emb)
        self.human = HumanFactorsNet_V3(emb=emb)

    def forward(self, spatial, emissions, coastal, temporal, short_term, grace_2d):
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal, grace_2d)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        return {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1)}

    def forward_with_embeddings(self, spatial, emissions, coastal, temporal, short_term, grace_2d):
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal, grace_2d)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        return ({'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1)}, {'spatial': zs, 'human': zh, 'temporal': zt, 'short': zr})

class SeaLevelModel_V6(SeaLevelModel_V3):
    """V3 (LayerNorm fusion) + 3 physics heads attached to specific branches.
    steric_head  reads from spatial  (where SST lives)
    dynamic_head reads from spatial  (where winds live)
    mass_head    reads from human    (where GRACE/runoff lives)"""

    def __init__(self, emb=EMBED_DIM):
        super().__init__(emb=emb)
        self.steric_head = PhysicsHead(emb=emb)
        self.dynamic_head = PhysicsHead(emb=emb)
        self.mass_head = PhysicsHead(emb=emb)

    def forward(self, spatial, emissions, coastal, temporal, short_term):
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        return {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh)}

    def forward_with_embeddings(self, spatial, emissions, coastal, temporal, short_term):
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        out = {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh)}
        embs = {'spatial': zs, 'human': zh, 'temporal': zt, 'short': zr}
        return (out, embs)

class SeaLevelModel_V7(SeaLevelModel_V6):
    """V6 (PINN heads) + GunpointFusion. Architecture forces human contribution."""

    def __init__(self, emb=EMBED_DIM, gate_floor=0.2):
        super().__init__(emb=emb)
        self.fuse = GunpointFusion(emb=emb, gate_floor=gate_floor)

class SeaLevelModel_V8(SeaLevelModel_V5):
    """V5 (FiLM human branch with GRACE 2D) + StrictGunpoint fusion + PINN heads."""

    def __init__(self, emb=EMBED_DIM, gate_floor=0.3):
        super().__init__(emb=emb)
        self.fuse = StrictGunpointFusion(emb=emb, gate_floor=gate_floor)
        self.steric_head = PhysicsHead(emb=emb)
        self.dynamic_head = PhysicsHead(emb=emb)
        self.mass_head = PhysicsHead(emb=emb)

    def forward(self, spatial, emissions, coastal, temporal, short_term, grace_2d):
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal, grace_2d)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        return {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh)}

    def forward_with_embeddings(self, spatial, emissions, coastal, temporal, short_term, grace_2d):
        (zs, ys) = self.spatial(spatial)
        (zh, yh) = self.human(emissions, coastal, grace_2d)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        out = {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh)}
        embs = {'spatial': zs, 'human': zh, 'temporal': zt, 'short': zr}
        return (out, embs)

class SeaLevelModel_V9(SeaLevelModel_V8):
    """V8 (Strict Gunpoint + PINN) with 5-channel anthropogenic human branch."""

    def __init__(self, emb=EMBED_DIM, gate_floor=0.3):
        super().__init__(emb=emb, gate_floor=gate_floor)
        self.human = HumanFactorsNet_V4(emb=emb, in_channels=5)

class SeaLevelModel_V11(SeaLevelModel_V9):

    def __init__(self, emb=EMBED_DIM, gate_floor=0.3):
        super().__init__(emb=emb, gate_floor=gate_floor)
        self.human = HumanFactorsNet_V5(emb=emb)

class SeaLevelModel_V13(SeaLevelModel_V9):
    """V9 fusion (StrictGunpoint + PINN heads) + HumanFactorsNet_V6 (Synthesizer).
    Spatial branch's prediction-equivalent is used as a V3.0 proxy for residual training.
    """

    def __init__(self, emb=EMBED_DIM, gate_floor=0.3):
        super().__init__(emb=emb, gate_floor=gate_floor)
        self.human = HumanFactorsNet_V6(emb=emb, in_channels=5)
        self.v30_proxy_head = nn.Sequential(nn.Linear(emb, emb // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(emb // 2, 1))

    def forward(self, spatial, emissions, coastal, temporal, short_term, anthro_5ch):
        (zs, ys) = self.spatial(spatial)
        (zh, yh, residual_pred) = self.human(emissions, coastal, anthro_5ch)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        v30_proxy = self.v30_proxy_head(zs)
        return {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh), 'residual_pred': residual_pred, 'v30_proxy': v30_proxy}

    def forward_with_embeddings(self, spatial, emissions, coastal, temporal, short_term, anthro_5ch):
        (zs, ys) = self.spatial(spatial)
        (zh, yh, residual_pred) = self.human(emissions, coastal, anthro_5ch)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu, log_var) = self.fuse(zs, zh, zt, zr)
        v30_proxy = self.v30_proxy_head(zs)
        out = {'mu': mu, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh), 'residual_pred': residual_pred, 'v30_proxy': v30_proxy}
        embs = {'spatial': zs, 'human': zh, 'temporal': zt, 'short': zr}
        return (out, embs)

class SeaLevelModel_V14(SeaLevelModel_V9):
    """V9 + HumanFactorsNet_V6 (Synthesizer) + ADDITIVE residual contribution to mu.
    Final prediction = fusion_mu + alpha * human_residual_pred
    where alpha is forced to stay above 0.30.
    """

    def __init__(self, emb=EMBED_DIM, gate_floor=0.3, residual_floor=0.3):
        super().__init__(emb=emb, gate_floor=gate_floor)
        self.human = HumanFactorsNet_V6(emb=emb, in_channels=5)
        self.residual_raw_gate = nn.Parameter(torch.tensor(0.5))
        self.residual_floor = residual_floor

    def _resid_alpha(self):
        return self.residual_floor + torch.sigmoid(self.residual_raw_gate)

    def forward(self, spatial, emissions, coastal, temporal, short_term, anthro_5ch):
        (zs, ys) = self.spatial(spatial)
        (zh, yh, residual_pred) = self.human(emissions, coastal, anthro_5ch)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu_fusion, log_var) = self.fuse(zs, zh, zt, zr)
        alpha = self._resid_alpha()
        mu_final = mu_fusion + alpha * residual_pred
        return {'mu': mu_final, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh), 'mu_fusion': mu_fusion, 'residual_pred': residual_pred, 'alpha': alpha.detach()}

    def forward_with_embeddings(self, spatial, emissions, coastal, temporal, short_term, anthro_5ch):
        (zs, ys) = self.spatial(spatial)
        (zh, yh, residual_pred) = self.human(emissions, coastal, anthro_5ch)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu_fusion, log_var) = self.fuse(zs, zh, zt, zr)
        alpha = self._resid_alpha()
        mu_final = mu_fusion + alpha * residual_pred
        out = {'mu': mu_final, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh), 'mu_fusion': mu_fusion, 'residual_pred': residual_pred, 'alpha': alpha.detach()}
        embs = {'spatial': zs, 'human': zh, 'temporal': zt, 'short': zr}
        return (out, embs)

class SeaLevelModel_V18(SeaLevelModel_V14):

    def __init__(self, emb=EMBED_DIM, gate_floor=0.3, residual_floor=0.3):
        super().__init__(emb=emb, gate_floor=gate_floor, residual_floor=residual_floor)
        self.human = HumanFactorsNet_V8(emb=emb)

    def forward(self, spatial, emissions, coastal, temporal, short_term, anthro_4ch, coastal_timelapse, ocean_chl, s1_sar):
        (zs, ys) = self.spatial(spatial)
        (zh, yh, residual_pred) = self.human(emissions, coastal, anthro_4ch, coastal_timelapse, ocean_chl, s1_sar)
        (zt, yt) = self.temporal(temporal)
        (zr, yr) = self.short(short_term)
        (mu_fusion, log_var) = self.fuse(zs, zh, zt, zr)
        alpha = self._resid_alpha()
        mu_final = mu_fusion + alpha * residual_pred
        return {'mu': mu_final, 'log_var': log_var, 'aux': torch.cat([ys, yh, yt, yr], dim=1), 'steric': self.steric_head(zs), 'dynamic': self.dynamic_head(zs), 'mass': self.mass_head(zh), 'mu_fusion': mu_fusion, 'residual_pred': residual_pred, 'alpha': alpha.detach()}

    def forward_with_embeddings(self, spatial, emissions, coastal, temporal, short_term, anthro_4ch, coastal_timelapse, ocean_chl, s1_sar):
        out = self.forward(spatial, emissions, coastal, temporal, short_term, anthro_4ch, coastal_timelapse, ocean_chl, s1_sar)
        (zs, _) = self.spatial(spatial)
        (zh, _, _) = self.human(emissions, coastal, anthro_4ch, coastal_timelapse, ocean_chl, s1_sar)
        (zt, _) = self.temporal(temporal)
        (zr, _) = self.short(short_term)
        return (out, {'spatial': zs, 'human': zh, 'temporal': zt, 'short': zr})
V_DOOM7 = SeaLevelModel_V18
if __name__ == '__main__':
    m = SeaLevelModel_V18()
    n = sum((p.numel() for p in m.parameters())) / 1000000.0
    print(f'V_DOOM7 (SeaLevelModel_V18) instantiated: {n:.2f} M params')