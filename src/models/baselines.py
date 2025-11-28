"""
Baseline Models for DimABSA Comparison

Includes:
1. TransformerVARegressor - Generic transformer baseline with simple pooling
2. BERTDimABSA - BERT-base baseline
3. RoBERTaDimABSA - RoBERTa-base baseline
4. SimplifiedDeBERTa - DeBERTa without attention mechanisms and lexicon

These baselines help demonstrate the value of:
- DeBERTa-V3 vs other transformer encoders
- Attention-weighted aspect pooling vs simple mean pooling
- Lexicon integration vs pure neural approach
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TransformerVARegressor(nn.Module):
    """
    Generic Transformer-based VA regressor.

    Uses [CLS] token representation or mean pooling for prediction.
    Serves as base class for BERT and RoBERTa baselines.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: str = "cls",  # "cls" or "mean"
        hidden_size: int = 256,
        dropout: float = 0.1,
        output_scaling: str = "sigmoid",
        va_min: float = 1.0,
        va_max: float = 9.0
    ):
        """
        Initialize transformer baseline.

        Args:
            model_name: HuggingFace model name/path
            pooling: Pooling strategy ("cls" or "mean")
            hidden_size: Hidden size for regression heads
            dropout: Dropout probability
            output_scaling: Output scaling method
            va_min: Minimum output value
            va_max: Maximum output value
        """
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.output_scaling = output_scaling
        self.va_min = va_min
        self.va_max = va_max
        self.va_range = va_max - va_min

        # Load transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_hidden = self.transformer.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Regression heads - simple single layer
        self.valence_head = nn.Sequential(
            nn.Linear(transformer_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(transformer_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def _scale_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply output scaling to ensure [va_min, va_max] range."""
        if self.output_scaling == "sigmoid":
            return self.va_min + self.va_range * torch.sigmoid(logits)
        elif self.output_scaling == "tanh":
            return self.va_min + self.va_range * (torch.tanh(logits / 2) + 1) / 2
        else:
            return torch.clamp(logits, self.va_min, self.va_max)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]

        Returns:
            Dictionary with 'valence' and 'arousal' predictions
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Pooling
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]  # [CLS] token
        else:
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask

        pooled = self.dropout(pooled)

        # Predictions
        valence_logits = self.valence_head(pooled).squeeze(-1)
        arousal_logits = self.arousal_head(pooled).squeeze(-1)

        valence = self._scale_output(valence_logits)
        arousal = self._scale_output(arousal_logits)

        return {
            'valence': valence,
            'arousal': arousal
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for prediction."""
        outputs = self.forward(input_ids, attention_mask, **kwargs)
        return outputs['valence'], outputs['arousal']

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class BERTDimABSA(TransformerVARegressor):
    """BERT-base baseline for DimABSA."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: str = "cls",
        hidden_size: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            pooling=pooling,
            hidden_size=hidden_size,
            dropout=dropout,
            **kwargs
        )


class RoBERTaDimABSA(TransformerVARegressor):
    """RoBERTa-base baseline for DimABSA."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        pooling: str = "cls",
        hidden_size: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            pooling=pooling,
            hidden_size=hidden_size,
            dropout=dropout,
            **kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,  # RoBERTa doesn't use this
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """RoBERTa doesn't use token_type_ids."""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,  # Explicitly set to None for RoBERTa
            **kwargs
        )


class SimplifiedDeBERTa(nn.Module):
    """
    Simplified DeBERTa baseline without attention mechanisms and lexicon.

    Uses mean pooling over aspect tokens instead of attention-weighted pooling.
    No cross-attention or lexicon features.
    Single-layer regression heads.

    This baseline helps demonstrate the value of:
    - Attention-weighted aspect pooling
    - Cross-attention mechanism
    - Lexicon integration
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        hidden_size: int = 256,
        dropout: float = 0.1,
        output_scaling: str = "sigmoid",
        va_min: float = 1.0,
        va_max: float = 9.0
    ):
        """
        Initialize simplified DeBERTa.

        Args:
            model_name: DeBERTa model name
            hidden_size: Hidden size for regression heads
            dropout: Dropout probability
            output_scaling: Output scaling method
            va_min: Minimum output value
            va_max: Maximum output value
        """
        super().__init__()

        self.output_scaling = output_scaling
        self.va_min = va_min
        self.va_max = va_max
        self.va_range = va_max - va_min

        # Load DeBERTa backbone
        self.deberta = AutoModel.from_pretrained(model_name)
        transformer_hidden = self.deberta.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Simple single-layer regression heads (like starter code)
        self.valence_head = nn.Linear(transformer_hidden, 1)
        self.arousal_head = nn.Linear(transformer_hidden, 1)

    def _scale_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply output scaling."""
        if self.output_scaling == "sigmoid":
            return self.va_min + self.va_range * torch.sigmoid(logits)
        elif self.output_scaling == "tanh":
            return self.va_min + self.va_range * (torch.tanh(logits / 2) + 1) / 2
        else:
            return torch.clamp(logits, self.va_min, self.va_max)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with simple mean pooling.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            aspect_mask: Optional aspect mask [batch, seq_len]
            token_type_ids: Optional token type IDs

        Returns:
            Dictionary with 'valence' and 'arousal' predictions
        """
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Simple pooling strategy
        if aspect_mask is not None and aspect_mask.sum() > 0:
            # Mean pooling over aspect tokens
            aspect_mask_expanded = aspect_mask.unsqueeze(-1)
            aspect_sum = (hidden_states * aspect_mask_expanded).sum(dim=1)
            aspect_count = aspect_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = aspect_sum / aspect_count
        else:
            # Fallback to [CLS] token
            pooled = hidden_states[:, 0, :]

        pooled = self.dropout(pooled)

        # Predictions
        valence_logits = self.valence_head(pooled).squeeze(-1)
        arousal_logits = self.arousal_head(pooled).squeeze(-1)

        valence = self._scale_output(valence_logits)
        arousal = self._scale_output(arousal_logits)

        return {
            'valence': valence,
            'arousal': arousal
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for prediction."""
        outputs = self.forward(input_ids, attention_mask, aspect_mask, **kwargs)
        return outputs['valence'], outputs['arousal']

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class AspectMeanPoolingModel(nn.Module):
    """
    Transformer model with mean pooling over aspect tokens.

    A middle-ground baseline between simple [CLS] pooling and
    full attention-weighted pooling. Uses the aspect mask but
    without learned attention weights.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        hidden_size: int = 256,
        num_head_layers: int = 2,
        dropout: float = 0.1,
        output_scaling: str = "sigmoid",
        va_min: float = 1.0,
        va_max: float = 9.0
    ):
        super().__init__()

        self.output_scaling = output_scaling
        self.va_min = va_min
        self.va_max = va_max
        self.va_range = va_max - va_min

        # Load transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_hidden = self.transformer.config.hidden_size

        # Multi-layer regression heads (like main model)
        self.valence_head = self._build_head(
            transformer_hidden, hidden_size, num_head_layers, dropout
        )
        self.arousal_head = self._build_head(
            transformer_hidden, hidden_size, num_head_layers, dropout
        )

    def _build_head(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ) -> nn.Sequential:
        """Build a multi-layer regression head."""
        layers = []
        current_size = input_size

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        return nn.Sequential(*layers)

    def _scale_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply output scaling."""
        if self.output_scaling == "sigmoid":
            return self.va_min + self.va_range * torch.sigmoid(logits)
        elif self.output_scaling == "tanh":
            return self.va_min + self.va_range * (torch.tanh(logits / 2) + 1) / 2
        else:
            return torch.clamp(logits, self.va_min, self.va_max)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with aspect mean pooling."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        # Mean pooling over aspect tokens
        aspect_mask_expanded = aspect_mask.unsqueeze(-1)
        aspect_sum = (hidden_states * aspect_mask_expanded).sum(dim=1)
        aspect_count = aspect_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = aspect_sum / aspect_count

        # Predictions
        valence_logits = self.valence_head(pooled).squeeze(-1)
        arousal_logits = self.arousal_head(pooled).squeeze(-1)

        valence = self._scale_output(valence_logits)
        arousal = self._scale_output(arousal_logits)

        return {
            'valence': valence,
            'arousal': arousal
        }

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def get_baseline_model(
    baseline_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        baseline_type: One of "bert", "roberta", "simplified", "mean_pooling"
        **kwargs: Model-specific arguments

    Returns:
        Initialized baseline model
    """
    baseline_type = baseline_type.lower()

    if baseline_type == "bert":
        return BERTDimABSA(**kwargs)
    elif baseline_type == "roberta":
        return RoBERTaDimABSA(**kwargs)
    elif baseline_type == "simplified" or baseline_type == "simplified_deberta":
        return SimplifiedDeBERTa(**kwargs)
    elif baseline_type == "mean_pooling":
        return AspectMeanPoolingModel(**kwargs)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


if __name__ == "__main__":
    print("Testing baseline models...")

    batch_size = 2
    seq_len = 64

    # Test data
    dummy_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'aspect_mask': torch.zeros(batch_size, seq_len),
    }
    dummy_input['aspect_mask'][:, 10:15] = 1

    # Test BERT baseline
    print("\n1. Testing BERT baseline...")
    bert_model = BERTDimABSA()
    print(f"   Parameters: {bert_model.get_num_params():,}")

    with torch.no_grad():
        outputs = bert_model(
            input_ids=dummy_input['input_ids'],
            attention_mask=dummy_input['attention_mask']
        )
    print(f"   Valence: {outputs['valence']}")
    print(f"   Arousal: {outputs['arousal']}")

    # Test Simplified DeBERTa
    print("\n2. Testing Simplified DeBERTa...")
    simple_model = SimplifiedDeBERTa()
    print(f"   Parameters: {simple_model.get_num_params():,}")

    with torch.no_grad():
        outputs = simple_model(**dummy_input)
    print(f"   Valence: {outputs['valence']}")
    print(f"   Arousal: {outputs['arousal']}")

    print("\nAll baseline tests passed!")
