from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dims=(256, 256, 256),
        dropout: float = 0.0,
        use_deltas: bool = True,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.use_deltas = use_deltas

        # per point features:
        # base (x,z) + mid (x,z) + half-width (x,z) = 6
        feat_per_point = 6
        if use_deltas:
            # deltas for mid and half-width (x,z each) = +4
            feat_per_point += 4  # -> 10 total when use_deltas=True

        self.feat_per_point = feat_per_point
        self.seq_len = 2 * n_track
        flat_in = self.seq_len * self.feat_per_point  # 20 * 10 = 200

        self.norm_in = nn.LayerNorm(flat_in)

        layers = []
        last = flat_in
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, n_waypoints * 2)]
        self.net = nn.Sequential(*layers)

    def _build_features(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        left/right: (B, n_track, 2) where [:, :, 0]=x (lateral), [:, :, 1]=z (longitudinal)
        Returns (B, 2*n_track, feat_per_point)
        """
        mid = 0.5 * (left + right)          # (B, n, 2)
        halfw = 0.5 * (right - left)        # (B, n, 2)

        base  = torch.cat([left, right], dim=1)   # (B, 2n, 2)
        mid2  = torch.cat([mid,  mid],  dim=1)    # (B, 2n, 2)
        half2 = torch.cat([halfw, halfw], dim=1)  # (B, 2n, 2)

        feats = [base, mid2, half2]               # 6 channels per point

        if self.use_deltas:
            def deltas(seq):
                d = seq[:, 1:, :] - seq[:, :-1, :]
                pad = torch.zeros(seq.size(0), 1, seq.size(2), device=seq.device, dtype=seq.dtype)
                return torch.cat([d, pad], dim=1)
            feats += [deltas(mid2), deltas(half2)]  # +4 channels

        return torch.cat(feats, dim=-1)           # (B, 2n, feat_per_point)

    def _mid_anchors(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Choose n_waypoints midline points as anchors (B, n_waypoints, 2).
        Indices are spaced along the n_track sequence.
        """
        B, n, _ = left.shape
        mid = 0.5 * (left + right)  # (B, n, 2)

        # evenly spaced integer indices in [0, n-1]
        # e.g., for n=10, n_waypoints=3 -> [0, 5, 9]
        idx_f = torch.linspace(0, n - 1, steps=self.n_waypoints, device=mid.device)
        idx = idx_f.round().long()  # (n_waypoints,)

        anchors = mid.index_select(dim=1, index=idx)  # (B, n_waypoints, 2)
        return anchors

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.size(0)
        seq = self._build_features(track_left, track_right)   # (B, 2n, F)
        flat = self.norm_in(seq.reshape(B, -1))               # (B, 2n*F)

        head = self.net(flat).view(B, self.n_waypoints, 2)    # residual head
        anchors = self._mid_anchors(track_left, track_right)  # (B, n_waypoints, 2)

        # Residualize ONLY x; keep z absolute
        pred_x = anchors[..., 0] + head[..., 0]               # residual around midline
        pred_z = head[..., 1]                                  # absolute z
        out = torch.stack([pred_x, pred_z], dim=-1)
        return out


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        add_side_token: bool = True,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.add_side_token = add_side_token  

        feat_per_point = 10  # 6 + 4
        if add_side_token:
            feat_per_point += 1

        self.input_proj = nn.Linear(feat_per_point, d_model)
        self.mem_pos = nn.Embedding(2 * n_track, d_model)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, S, E)
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)  # (x, z) per waypoint

        # optional normalization on inputs
        self.input_norm = nn.LayerNorm(d_model)

    def _build_memory(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        """
        Concatenate L/R tracks into a single sequence and (optionally) add a side token.
        Args:
            track_left, track_right: (B, n_track, 2)
        Returns:
            mem: (B, 2*n_track, d_model)
        """
        B, n, _ = track_left.shape

        # mid and half-width in ego BEV
        mid   = 0.5 * (track_left + track_right)      # (B, n, 2)
        halfw = 0.5 * (track_right - track_left)      # (B, n, 2)

        # concatenate L/R to match 2n sequence
        base  = torch.cat([track_left, track_right], dim=1)   # (B, 2n, 2)
        mid2  = torch.cat([mid, mid], dim=1)                  # (B, 2n, 2)
        half2 = torch.cat([halfw, halfw], dim=1)              # (B, 2n, 2)

        # deltas along the sequence (like your MLP)
        def deltas(seq):
            d   = seq[:, 1:, :] - seq[:, :-1, :]
            pad = torch.zeros(seq.size(0), 1, seq.size(2), device=seq.device, dtype=seq.dtype)
            return torch.cat([d, pad], dim=1)

        d_mid   = deltas(mid2)    # (B, 2n, 2)
        d_half  = deltas(half2)   # (B, 2n, 2)

        feats = [base, mid2, half2, d_mid, d_half]    # -> 10 dims per point

        if self.add_side_token:
            side_left  = torch.zeros(B, n, 1, device=track_left.device, dtype=track_left.dtype)
            side_right = torch.ones(B, n, 1, device=track_left.device, dtype=track_left.dtype)
            side = torch.cat([side_left, side_right], dim=1)   # (B, 2n, 1)
            feats.append(side)

        mem_in = torch.cat(feats, dim=-1)                      # (B, 2n, 10 or 11)
        mem = self.input_proj(mem_in)                          # (B, 2n, d_model)

        # learned positional embeddings
        pos_ids = torch.arange(mem.size(1), device=mem.device)[None, :]  # (1, 2n)
        mem = mem + self.mem_pos(pos_ids)

        mem = self.input_norm(mem)
        return mem

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.size(0)
        mem = self._build_memory(track_left, track_right)
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        dec = self.decoder(q, mem)
        head = self.head(dec)  # (B, n_wp, 2): [dx_residual, dz_delta_raw]

        mid = 0.5 * (track_left + track_right)
        idx = torch.linspace(0, self.n_track-1, steps=self.n_waypoints, device=mid.device).round().long()
        anchors = mid.index_select(dim=1, index=idx)             # (B, n_wp, 2)

        pred_x = anchors[..., 0] + head[..., 0]                  # residual x
        # enforce positive forward motion
        dz = torch.nn.functional.softplus(head[..., 1])          # (B, n_wp)
        pred_z = torch.cumsum(dz, dim=1)                         # monotonic z

        out = torch.stack([pred_x, pred_z], dim=-1)
        return out


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = (image - self.input_mean[None,:,None,None]) / self.input_std[None,:,None,None]
        x = self.backbone(x).flatten(1)           # (B,128)
        out = self.head(x).view(-1, self.n_waypoints, 2)  # (B,n,2)

        x_abs      = out[..., 0]                      # predict absolute x
        z_delta_sp = torch.nn.functional.softplus(out[..., 1])  # positive deltas
        z_abs      = torch.cumsum(z_delta_sp, dim=1)  # monotonic forward

        return torch.stack([x_abs, z_abs], dim=-1)    # (B,n,2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
