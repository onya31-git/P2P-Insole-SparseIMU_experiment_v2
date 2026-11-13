import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal, causal-safe)
# ---------------------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    """標準的なサイン/コサイン位置エンコーディング。

    Parameters
    ----------
    d_model : int
        埋め込み次元。
    max_len : int
        最大シーケンス長。
    dropout : float
        ドロップアウト率。
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                      # [max_len, d_model]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                                     # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, T, d_model]

        Returns
        -------
        Tensor  [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# LowerBodyNet (Transformer版)
# ---------------------------------------------------------------------------
class LowerBodyNet(nn.Module):
    """下半身スケルタル潜在表現を推定するネットワーク。

    時系列処理に Transformer Encoder を使用する。

    Parameters
    ----------
    feature_dim : int
        各センサーエンコーダの出力次元。
    body_param_dim : int
        身体パラメータの次元数。
    hidden_dim : int
        Transformer の d_model（= 出力次元）。
    num_layers : int
        Transformer Encoder の層数。
    nhead : int
        Multi-Head Attention のヘッド数。d_model の約数であること。
    ffn_dim : int
        Feedforward Network の中間次元。None の場合 hidden_dim * 4。
    dropout : float
        ドロップアウト率。
    causal : bool
        True の場合、因果マスク（未来を参照しない）を適用する。
        リアルタイム推論時は True を推奨。
    max_seq_len : int
        位置エンコーディングの最大シーケンス長。
    """
    def __init__(
        self,
        feature_dim: int = 128,
        body_param_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
        causal: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # 身体パラメータ埋め込み                                               #
        # ------------------------------------------------------------------ #
        self.body_param_embed = nn.Sequential(
            nn.Linear(body_param_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # ------------------------------------------------------------------ #
        # 入力射影                                                             #
        # input_dim = LeftFoot + RightFoot + Pelvis + BodyParam = 4 * feature_dim
        # ------------------------------------------------------------------ #
        input_dim = feature_dim * 4
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ------------------------------------------------------------------ #
        # 位置エンコーディング                                                  #
        # ------------------------------------------------------------------ #
        self.pos_enc = _PositionalEncoding(
            d_model=hidden_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # ------------------------------------------------------------------ #
        # Transformer Encoder                                                  #
        # ------------------------------------------------------------------ #
        ffn_dim = ffn_dim or hidden_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,      # [B, T, d_model]
            norm_first=True,       # Pre-LN: 学習安定性向上
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ------------------------------------------------------------------ #
        # 出力射影                                                              #
        # ------------------------------------------------------------------ #
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ------------------------------------------------------------------ #
        # オプション                                                             #
        # ------------------------------------------------------------------ #
        self.causal = causal

    # ---------------------------------------------------------------------- #
    # ヘルパー: 因果マスク生成                                                  #
    # ---------------------------------------------------------------------- #
    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """上三角が -inf の因果マスク [T, T] を返す。"""
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device), diagonal=1
        )
        return mask

    # ---------------------------------------------------------------------- #
    # Forward (Transformer版)                                                  #
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        left_foot: torch.Tensor,
        right_foot: torch.Tensor,
        pelvis: torch.Tensor,
        body_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        left_foot, right_foot, pelvis : Tensor  [B, T, feature_dim]
        body_params : Tensor  [B, body_param_dim]

        Returns
        -------
        lower_body_latent : Tensor  [B, T, hidden_dim]
        """
        B, T, _ = left_foot.shape

        # 身体パラメータを時系列方向に拡張
        bp_embed = self.body_param_embed(body_params)           # [B, feature_dim]
        bp_embed = bp_embed.unsqueeze(1).expand(-1, T, -1)      # [B, T, feature_dim]

        # 特徴量を結合 → 射影
        x = torch.cat([left_foot, right_foot, pelvis, bp_embed], dim=-1)  # [B, T, input_dim]
        x = self.input_proj(x)                                  # [B, T, hidden_dim]

        # 位置エンコーディング
        x = self.pos_enc(x)                                     # [B, T, hidden_dim]

        # 因果マスク
        mask = self._causal_mask(T, x.device) if self.causal else None

        # Transformer Encoder
        x = self.transformer(x, mask=mask)                      # [B, T, hidden_dim]

        # 出力射影
        lower_body_latent = self.fc_out(x)                      # [B, T, hidden_dim]

        return lower_body_latent


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    B, T, F_dim, BP_dim = 2, 64, 128, 8

    # --- Transformer版 ---
    model_tf = LowerBodyNet(
        feature_dim=F_dim,
        body_param_dim=BP_dim,
        hidden_dim=256,
        num_layers=2,
        nhead=8,
        causal=False,
    )
    lf = torch.randn(B, T, F_dim)
    rf = torch.randn(B, T, F_dim)
    p  = torch.randn(B, T, F_dim)
    bp = torch.randn(B, BP_dim)

    latent_tf = model_tf(lf, rf, p, bp)
    print("Transformer LowerBodyLatent shape:", latent_tf.shape)   # [2, 64, 256]

    # --- 因果マスク版 ---
    model_causal = LowerBodyNet(
        feature_dim=F_dim,
        body_param_dim=BP_dim,
        causal=True,
    )
    latent_causal = model_causal(lf, rf, p, bp)
    print("Causal Transformer LowerBodyLatent shape:", latent_causal.shape)
