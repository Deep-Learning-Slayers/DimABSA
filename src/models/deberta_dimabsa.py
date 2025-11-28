"""
DeBERTa-based Model for Dimensional Aspect-Based Sentiment Analysis

Main model architecture featuring:
- DeBERTa-V3-base backbone
- Attention-weighted aspect representation
- Cross-attention for aspect-context interaction
- Lexicon feature integration
- Dual regression heads for valence and arousal
- Sigmoid output scaling for [1,9] range
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model, DebertaV2Config, AutoModel


@dataclass
class DeBERTaDimABSAConfig:
    """Configuration for DeBERTaDimABSA model."""

    # Model backbone
    model_name: str = "microsoft/deberta-v3-base"
    hidden_size: int = 768

    # Attention mechanism
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Regression heads
    head_hidden_size: int = 256
    head_dropout: float = 0.1
    num_head_layers: int = 2

    # Lexicon integration
    use_lexicon: bool = True
    lexicon_feature_dim: int = 8

    # Output scaling
    output_scaling: str = "sigmoid"  # "sigmoid", "tanh", or "linear"
    va_min: float = 1.0
    va_max: float = 9.0

    # Training
    freeze_backbone_layers: int = 0  # Number of bottom layers to freeze


class AspectAttention(nn.Module):
    """
    Attention mechanism for computing weighted aspect representation.

    Computes attention weights based on the relationship between
    each token and the aspect representation, then performs
    weighted pooling over aspect tokens.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size

        # Attention weight computation
        # W_a · [h_i; h_aspect] -> scalar attention score
        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention-weighted aspect representation.

        Args:
            hidden_states: Token representations [batch, seq_len, hidden]
            aspect_mask: Binary mask for aspect tokens [batch, seq_len]
            attention_mask: Optional mask for valid tokens [batch, seq_len]

        Returns:
            Aspect representation [batch, hidden]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute initial aspect representation (mean of aspect tokens)
        aspect_mask_expanded = aspect_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        aspect_sum = (hidden_states * aspect_mask_expanded).sum(dim=1)
        aspect_count = aspect_mask.sum(dim=1, keepdim=True).clamp(min=1)
        aspect_mean = aspect_sum / aspect_count  # [batch, hidden]

        # Expand aspect_mean to match sequence length
        aspect_expanded = aspect_mean.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate token representations with aspect representation
        concat_features = torch.cat([hidden_states, aspect_expanded], dim=-1)

        # Compute attention scores
        attention_hidden = torch.tanh(self.attention_linear(concat_features))
        attention_scores = self.attention_score(attention_hidden).squeeze(-1)

        # Apply aspect mask (only attend to aspect tokens)
        # Set non-aspect tokens to large negative value
        attention_scores = attention_scores.masked_fill(
            aspect_mask == 0, float('-inf')
        )

        # Softmax over aspect tokens
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Handle cases where all attention weights are -inf (no aspect tokens)
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )

        # Weighted sum of hidden states
        weighted_aspect = torch.bmm(
            attention_weights.unsqueeze(1),
            hidden_states
        ).squeeze(1)

        # Layer norm and residual connection with mean pooling
        output = self.layer_norm(weighted_aspect + aspect_mean)

        return output


class CrossAttention(nn.Module):
    """
    Cross-attention layer for aspect-context interaction.

    Allows the aspect representation to attend to context tokens,
    capturing relevant sentiment information from the surrounding text.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        aspect_repr: torch.Tensor,
        context_states: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention from aspect to context.

        Args:
            aspect_repr: Aspect representation [batch, hidden]
            context_states: Context token representations [batch, seq_len, hidden]
            context_mask: Mask for context tokens [batch, seq_len]

        Returns:
            Enhanced aspect representation [batch, hidden]
        """
        # Expand aspect to [batch, 1, hidden] for attention
        aspect_query = aspect_repr.unsqueeze(1)

        # Create key padding mask for attention
        key_padding_mask = None
        if context_mask is not None:
            # True = ignore position, False = attend
            key_padding_mask = (context_mask == 0)

        # Cross-attention: aspect attends to context
        attended, _ = self.multihead_attention(
            query=aspect_query,
            key=context_states,
            value=context_states,
            key_padding_mask=key_padding_mask
        )

        # Residual connection and layer norm
        attended = attended.squeeze(1)  # [batch, hidden]
        aspect_repr = self.layer_norm1(aspect_repr + attended)

        # Feed-forward with residual
        ff_output = self.feed_forward(aspect_repr)
        output = self.layer_norm2(aspect_repr + ff_output)

        return output


class RegressionHead(nn.Module):
    """
    Multi-layer regression head for predicting V or A scores.

    Supports configurable number of hidden layers and output scaling.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_scaling: str = "sigmoid",
        output_min: float = 1.0,
        output_max: float = 9.0
    ):
        super().__init__()

        self.output_scaling = output_scaling
        self.output_min = output_min
        self.output_max = output_max
        self.output_range = output_max - output_min

        layers = []
        current_size = input_size

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(current_size, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with output scaling.

        Args:
            x: Input features [batch, input_size]

        Returns:
            Scaled output [batch] in [output_min, output_max] range
        """
        logits = self.layers(x).squeeze(-1)

        if self.output_scaling == "sigmoid":
            # output = min + range * sigmoid(logit)
            # Ensures smooth output in [1, 9]
            output = self.output_min + self.output_range * torch.sigmoid(logits)

        elif self.output_scaling == "tanh":
            # output = min + range * (tanh(logit/2) + 1) / 2
            output = self.output_min + self.output_range * (torch.tanh(logits / 2) + 1) / 2

        elif self.output_scaling == "linear":
            # Linear clipping
            output = torch.clamp(logits, self.output_min, self.output_max)

        else:
            raise ValueError(f"Unknown output scaling: {self.output_scaling}")

        return output


class DeBERTaDimABSA(nn.Module):
    """
    DeBERTa-based model for Dimensional Aspect-Based Sentiment Analysis.

    Architecture:
    1. DeBERTa-V3 backbone encodes input "[CLS] text [SEP] aspect [SEP]"
    2. Aspect attention computes weighted aspect representation
    3. Cross-attention enhances aspect with context information
    4. Optional lexicon features are concatenated
    5. Dual regression heads predict valence and arousal
    """

    def __init__(self, config: Optional[DeBERTaDimABSAConfig] = None):
        super().__init__()

        if config is None:
            config = DeBERTaDimABSAConfig()

        self.config = config

        # Load DeBERTa backbone
        self.deberta = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.deberta.config.hidden_size

        # Freeze bottom layers if specified
        if config.freeze_backbone_layers > 0:
            self._freeze_backbone_layers(config.freeze_backbone_layers)

        # Aspect attention
        self.aspect_attention = AspectAttention(
            hidden_size=hidden_size,
            dropout=config.attention_dropout
        )

        # Cross-attention for aspect-context interaction
        self.cross_attention = CrossAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout
        )

        # Lexicon feature projection
        self.use_lexicon = config.use_lexicon
        if self.use_lexicon:
            self.lexicon_projection = nn.Linear(
                config.lexicon_feature_dim,
                config.head_hidden_size
            )
            regression_input_size = hidden_size + config.head_hidden_size
        else:
            self.lexicon_projection = None
            regression_input_size = hidden_size

        # Regression heads
        self.valence_head = RegressionHead(
            input_size=regression_input_size,
            hidden_size=config.head_hidden_size,
            num_layers=config.num_head_layers,
            dropout=config.head_dropout,
            output_scaling=config.output_scaling,
            output_min=config.va_min,
            output_max=config.va_max
        )

        self.arousal_head = RegressionHead(
            input_size=regression_input_size,
            hidden_size=config.head_hidden_size,
            num_layers=config.num_head_layers,
            dropout=config.head_dropout,
            output_scaling=config.output_scaling,
            output_min=config.va_min,
            output_max=config.va_max
        )

    def _freeze_backbone_layers(self, num_layers: int):
        """Freeze the bottom N encoder layers."""
        # Freeze embeddings
        for param in self.deberta.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified encoder layers
        if hasattr(self.deberta, 'encoder'):
            for layer_idx in range(num_layers):
                if layer_idx < len(self.deberta.encoder.layer):
                    for param in self.deberta.encoder.layer[layer_idx].parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        lexicon_features: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VA prediction.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            aspect_mask: Mask indicating aspect tokens [batch, seq_len]
            lexicon_features: Optional lexicon features [batch, 8]
            token_type_ids: Optional token type IDs [batch, seq_len]

        Returns:
            Dictionary with 'valence' and 'arousal' predictions [batch]
        """
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Compute aspect representation with attention
        aspect_repr = self.aspect_attention(
            hidden_states=hidden_states,
            aspect_mask=aspect_mask,
            attention_mask=attention_mask
        )

        # Apply cross-attention for aspect-context interaction
        aspect_repr = self.cross_attention(
            aspect_repr=aspect_repr,
            context_states=hidden_states,
            context_mask=attention_mask
        )

        # Concatenate lexicon features if available
        if self.use_lexicon and lexicon_features is not None:
            lexicon_projected = self.lexicon_projection(lexicon_features)
            combined = torch.cat([aspect_repr, lexicon_projected], dim=-1)
        else:
            combined = aspect_repr

        # Predict valence and arousal
        valence = self.valence_head(combined)
        arousal = self.arousal_head(combined)

        return {
            'valence': valence,
            'arousal': arousal,
            'aspect_repr': aspect_repr  # For analysis/visualization
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        lexicon_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for prediction.

        Returns:
            Tuple of (valence, arousal) tensors
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_mask=aspect_mask,
            lexicon_features=lexicon_features,
            **kwargs
        )
        return outputs['valence'], outputs['arousal']

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[DeBERTaDimABSAConfig] = None):
        """Load model from saved checkpoint."""
        if config is None:
            config = DeBERTaDimABSAConfig()

        model = cls(config)
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save(self.state_dict(), path)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test model initialization and forward pass
    print("Testing DeBERTaDimABSA model...")

    config = DeBERTaDimABSAConfig(
        model_name="microsoft/deberta-v3-base",
        use_lexicon=True,
        output_scaling="sigmoid"
    )

    print(f"Config: {config}")

    # Create model
    model = DeBERTaDimABSA(config)
    print(f"Model parameters: {model.get_num_params():,} trainable")

    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 64

    dummy_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'aspect_mask': torch.zeros(batch_size, seq_len),
        'lexicon_features': torch.randn(batch_size, 8)
    }
    # Set some tokens as aspect
    dummy_input['aspect_mask'][:, 10:15] = 1

    print("\nForward pass...")
    with torch.no_grad():
        outputs = model(**dummy_input)

    print(f"Valence: {outputs['valence']}")
    print(f"Arousal: {outputs['arousal']}")
    print(f"Output range: V=[{outputs['valence'].min():.2f}, {outputs['valence'].max():.2f}], "
          f"A=[{outputs['arousal'].min():.2f}, {outputs['arousal'].max():.2f}]")
