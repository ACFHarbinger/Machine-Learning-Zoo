"""
Neuro-Symbolic Methods.

Hybrid architectures combining neural networks with symbolic reasoning,
enabling models that can learn from data while respecting logical constraints
and producing interpretable symbolic outputs.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from ..configs.wrappers import NeuroSymbolicConfig

logger = logging.getLogger(__name__)

__all__ = [
    "NeuroSymbolicOutput",
    "RuleEncoder",
    "SymbolicReasoner",
    "NeuroSymbolicNetwork",
    "NeuroSymbolicConfig",
    "LogicProgramExecutor",
]


class NeuroSymbolicOutput(NamedTuple):
    """Output from a neuro-symbolic model."""

    prediction: torch.Tensor  # Final integrated prediction
    neural_output: torch.Tensor  # Raw neural network output
    symbolic_output: torch.Tensor  # Symbolic reasoning output
    rule_attention: torch.Tensor  # Attention weights over rules
    confidence: torch.Tensor  # Confidence score for the prediction


class RuleEncoder(nn.Module):
    """
    Encodes symbolic rules as differentiable embeddings.

    Each rule is represented as a learnable embedding vector that captures
    the logical structure of if-then relationships. Rules are applied to
    input features via attention-weighted combination.
    """

    def __init__(
        self,
        num_rules: int,
        rule_dim: int,
        input_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_rules = num_rules
        self.rule_dim = rule_dim

        # Learnable rule embeddings: each row is a rule
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, rule_dim) * 0.02)

        # Project input features to rule space for attention
        self.query_proj = nn.Linear(input_dim, rule_dim)

        # Rule condition and conclusion networks
        self.condition_net = nn.Sequential(
            nn.Linear(rule_dim, rule_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rule_dim, rule_dim),
        )
        self.conclusion_net = nn.Sequential(
            nn.Linear(rule_dim, rule_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rule_dim, rule_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rules to input features.

        Args:
            features: Input features of shape (batch, input_dim).
            temperature: Softmax temperature for rule selection.

        Returns:
            Tuple of (rule_output, rule_attention) where rule_output has
            shape (batch, rule_dim) and rule_attention has shape
            (batch, num_rules).
        """
        # Project features into rule space
        query = self.query_proj(features)  # (batch, rule_dim)

        # Compute attention over rules
        # (batch, rule_dim) @ (rule_dim, num_rules) -> (batch, num_rules)
        attn_logits = torch.matmul(query, self.rule_embeddings.t())
        attn_logits = attn_logits / (self.rule_dim**0.5)
        rule_attention = F.softmax(attn_logits / temperature, dim=-1)

        # Evaluate rule conditions
        conditions = self.condition_net(self.rule_embeddings)  # (num_rules, rule_dim)

        # Compute gated conclusions based on condition satisfaction
        # Condition satisfaction: how well input matches each rule's condition
        condition_scores = torch.sigmoid(torch.matmul(query, conditions.t()))  # (batch, num_rules)

        # Weight conclusions by both attention and condition satisfaction
        effective_weights = rule_attention * condition_scores
        effective_weights = effective_weights / (effective_weights.sum(dim=-1, keepdim=True) + 1e-8)

        conclusions = self.conclusion_net(self.rule_embeddings)  # (num_rules, rule_dim)

        # Weighted combination of rule conclusions
        rule_output = torch.matmul(effective_weights, conclusions)  # (batch, rule_dim)

        return rule_output, rule_attention


class LogicProgramExecutor(nn.Module):
    """
    Differentiable logic program executor.

    Implements forward chaining over a set of learned predicates,
    using soft unification to match rule antecedents against known facts.
    """

    def __init__(
        self,
        num_predicates: int,
        predicate_dim: int,
        num_steps: int = 3,
    ) -> None:
        super().__init__()
        self.num_predicates = num_predicates
        self.predicate_dim = predicate_dim
        self.num_steps = num_steps

        # Predicate embeddings
        self.predicate_embeddings = nn.Parameter(torch.randn(num_predicates, predicate_dim) * 0.02)

        # Implication weights: how strongly predicate i implies predicate j
        self.implication_net = nn.Sequential(
            nn.Linear(predicate_dim * 2, predicate_dim),
            nn.ReLU(),
            nn.Linear(predicate_dim, 1),
        )

        # Grounding network: maps input features to initial predicate truth values
        self.grounding = nn.Linear(predicate_dim, num_predicates)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Execute forward chaining on input features.

        Args:
            features: Input features of shape (batch, predicate_dim).

        Returns:
            Final predicate truth values of shape (batch, num_predicates).
        """
        # Ground predicates from input features
        truth_values = torch.sigmoid(self.grounding(features))  # (batch, num_predicates)

        # Build implication matrix
        # For each pair (i, j), compute implication strength
        pred_i = self.predicate_embeddings.unsqueeze(1).expand(-1, self.num_predicates, -1)  # (P, P, dim)
        pred_j = self.predicate_embeddings.unsqueeze(0).expand(self.num_predicates, -1, -1)  # (P, P, dim)
        pairs = torch.cat([pred_i, pred_j], dim=-1)  # (P, P, 2*dim)
        impl_matrix = torch.sigmoid(self.implication_net(pairs).squeeze(-1))  # (P, P)

        # Forward chaining: iteratively derive new facts
        for _ in range(self.num_steps):
            # For each predicate j, compute max implied truth value
            # implied_j = max_i(truth_i * impl(i->j))
            implied = torch.matmul(truth_values, impl_matrix)  # (batch, num_predicates)
            # Soft OR: combine existing truth values with newly derived ones
            truth_values = torch.max(truth_values, torch.sigmoid(implied))

        return truth_values


class SymbolicReasoner(nn.Module):
    """
    Combines rule-based reasoning with logic program execution.

    Processes neural features through both a rule encoder and a logic
    program executor, then fuses their outputs.
    """

    def __init__(self, config: NeuroSymbolicConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        self.rule_encoder = RuleEncoder(
            num_rules=config.num_rules,
            rule_dim=config.rule_dim,
            input_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        self.logic_executor = LogicProgramExecutor(
            num_predicates=config.num_predicates,
            predicate_dim=config.hidden_dim,
            num_steps=config.symbolic_depth,
        )

        # Fuse rule and logic outputs
        fuse_input_dim = config.rule_dim + config.num_predicates
        self.fusion = nn.Sequential(
            nn.Linear(fuse_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform symbolic reasoning over neural features.

        Args:
            features: Input features of shape (batch, input_dim).

        Returns:
            Tuple of (symbolic_output, rule_attention) where
            symbolic_output has shape (batch, hidden_dim).
        """
        h = self.input_proj(features)

        rule_output, rule_attention = self.rule_encoder(h, temperature=self.config.temperature)
        logic_output = self.logic_executor(h)

        fused = self.fusion(torch.cat([rule_output, logic_output], dim=-1))

        return fused, rule_attention


class NeuroSymbolicNetwork(nn.Module):
    """
    Full neuro-symbolic network combining a neural pathway with
    symbolic reasoning, integrated via a learned gating mechanism.

    The neural pathway captures pattern recognition from data while
    the symbolic pathway enforces logical consistency. The integration
    mode controls how these two streams are combined:

    - ``gated``: A learned gate balances neural vs. symbolic contributions.
    - ``residual``: Symbolic output is added as a residual to neural output.
    - ``attention``: Cross-attention between neural and symbolic representations.
    """

    def __init__(self, config: NeuroSymbolicConfig) -> None:
        super().__init__()
        self.config = config

        # Neural pathway
        self.neural_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Symbolic pathway
        self.symbolic_reasoner = SymbolicReasoner(config)

        # Integration mechanism
        if config.integration_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.Sigmoid(),
            )
        elif config.integration_mode == "attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True,
            )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> NeuroSymbolicOutput:
        """
        Forward pass through both neural and symbolic pathways.

        Args:
            x: Input tensor of shape (batch, input_dim) or (batch, seq, input_dim).
            **kwargs: Additional arguments (unused).

        Returns:
            NeuroSymbolicOutput with predictions, component outputs, and confidence.
        """
        # Pool sequence dimension if present
        if x.dim() == 3:
            x = x.mean(dim=1)

        # Neural pathway
        neural_features = self.neural_encoder(x)
        neural_output = self.output_proj(neural_features)

        # Symbolic pathway
        symbolic_features, rule_attention = self.symbolic_reasoner(x)
        symbolic_output = self.output_proj(symbolic_features)

        # Integration
        integrated = self._integrate(neural_features, symbolic_features)
        prediction = self.output_proj(integrated)

        # Confidence based on agreement between neural and symbolic
        confidence = self.confidence_head(integrated)

        return NeuroSymbolicOutput(
            prediction=prediction,
            neural_output=neural_output,
            symbolic_output=symbolic_output,
            rule_attention=rule_attention,
            confidence=confidence.squeeze(-1),
        )

    def _integrate(
        self,
        neural: torch.Tensor,
        symbolic: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate neural and symbolic representations."""
        mode = self.config.integration_mode

        if mode == "gated":
            gate_input = torch.cat([neural, symbolic], dim=-1)
            gate = self.gate(gate_input)
            return gate * neural + (1 - gate) * symbolic

        elif mode == "residual":
            return neural + symbolic

        elif mode == "attention":
            # Treat each as a single-token sequence for cross-attention
            neural_seq = neural.unsqueeze(1)  # (batch, 1, hidden)
            symbolic_seq = symbolic.unsqueeze(1)  # (batch, 1, hidden)
            attn_out, _ = self.cross_attn(
                query=neural_seq,
                key=symbolic_seq,
                value=symbolic_seq,
            )
            return attn_out.squeeze(1)

        else:
            raise ValueError(f"Unknown integration mode: {mode}")

    def get_rule_importance(self) -> torch.Tensor:
        """
        Return the learned rule embeddings for inspection.

        Returns:
            Rule embedding matrix of shape (num_rules, rule_dim).
        """
        return self.symbolic_reasoner.rule_encoder.rule_embeddings.data.clone()

    def symbolic_loss(
        self,
        predictions: torch.Tensor,
        constraints: list[tuple[int, int, str]],
    ) -> torch.Tensor:
        """
        Compute a penalty for violating symbolic constraints.

        Each constraint is a tuple ``(class_i, class_j, relation)`` where
        ``relation`` is one of ``"mutex"`` (mutually exclusive) or
        ``"implies"`` (class_i implies class_j).

        Args:
            predictions: Predicted probabilities of shape (batch, num_classes).
            constraints: List of symbolic constraint tuples.

        Returns:
            Scalar penalty loss.
        """
        probs = F.softmax(predictions, dim=-1)
        penalty = torch.tensor(0.0, device=predictions.device)

        for i, j, relation in constraints:
            if relation == "mutex":
                # Mutual exclusion: P(i) * P(j) should be 0
                penalty = penalty + (probs[:, i] * probs[:, j]).mean()
            elif relation == "implies":
                # Implication: P(i) <= P(j), penalize when P(i) > P(j)
                violation = F.relu(probs[:, i] - probs[:, j])
                penalty = penalty + violation.mean()

        return penalty
