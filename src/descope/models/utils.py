import torch
from torch import nn
from wppkg import NoRoPE
from typing import Optional
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from torch.distributions import NegativeBinomial, Bernoulli
from transformers.models.llama import LlamaModel, LlamaConfig


@dataclass
class DeSCOPEForATACOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


@dataclass
class DeSCOPEForRNAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


def kl_loss_func(
    prior_mean: torch.Tensor, 
    prior_logvar: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor
) -> torch.Tensor:
    kl_loss = -0.5 * torch.sum(
        1
        + posterior_logvar
        - prior_logvar
        - (posterior_logvar.exp() + (posterior_mean - prior_mean).pow(2)) / prior_logvar.exp(),
        dim=-1
    )
    return kl_loss.mean()


class ZINBLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x, mu, theta, pi) -> torch.Tensor:
        x = x.float()
        mu = mu.float()
        theta = theta.float()
        pi = pi.float()
        
        # Clamp values for numerical stability
        mu = torch.clamp(mu, min=self.eps, max=1e4)
        theta = torch.clamp(theta, min=self.eps, max=1e4)
        pi = torch.clamp(pi, min=self.eps, max=1-self.eps)

        # NB log-likelihood
        t1 = torch.lgamma(theta) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta)
        t2 = (theta + x) * torch.log1p(mu / theta) + x * (torch.log(theta) - torch.log(mu))
        nb_loss = t1 + t2

        # ZINB log-likelihood
        zero_mask = (x < self.eps).float()
        zinb_loss = -torch.log(
            pi + (1.0 - pi) * torch.exp(-nb_loss)
        ) * zero_mask - (1.0 - zero_mask) * (torch.log(1.0 - pi + self.eps) - nb_loss)

        return zinb_loss.mean()

    @torch.no_grad()
    def sample(self, mu, theta, pi) -> torch.Tensor:
        mu = mu.float()
        theta = theta.float()
        pi = pi.float()

        # Ensure numerical stability
        mu = torch.clamp(mu, min=self.eps, max=1e4)
        theta = torch.clamp(theta, min=self.eps, max=1e4)
        pi = torch.clamp(pi, min=self.eps, max=1-self.eps)

        # Compute NB parameters
        probs = mu / (theta + mu)
        total_count = theta

        # Sample from NB
        probs = torch.clamp(probs, min=0.0, max=1-self.eps)
        nb_dist = NegativeBinomial(total_count=total_count, probs=probs)
        nb_sample = nb_dist.sample()

        # Sample from Bernoulli for dropout
        bernoulli_dist = Bernoulli(probs=1 - pi)
        keep_mask = bernoulli_dist.sample()

        # Apply zero-inflation
        zinb_sample = nb_sample * keep_mask

        return zinb_sample


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU(),
    add_layernorm: bool = False
) -> nn.Sequential:
    """
    Build an MLP of `n_layers` from `in_dim` to `out_dim`.
    """
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        if add_layernorm:
            layers.append(nn.LayerNorm(out_dim))
    else:
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        if add_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation)  # instantiate the class
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if add_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation)  # instantiate again
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.rotary_emb = NoRoPE(
            head_dim=config.head_dim,
        )
        
        # Explicitly disable causal attention
        self.config.is_causal = False
        # force every layer to be non-causal
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever "attention_mask" the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):
        flash_attn_kwargs["is_causal"] = False
        
        # If no attention_mask is provided, create an all-ones mask (no masking)
        # This ensures bidirectional attention with correct device/dtype
        if attention_mask is None:
            # Get batch size (B) and sequence length (S) from input_embeds if available, else from input_ids.
            # If neither is available, fall back to attention_mask=None and log a warning.
            B = None
            S = None
            if inputs_embeds is not None:
                B, S = inputs_embeds.size(0), inputs_embeds.size(1)
            if B and S:
                attention_mask = torch.ones((B, 1, S, S), dtype=torch.float, device=inputs_embeds.device)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )


def get_bidirectional_llama_backbone(config: LlamaConfig) -> PreTrainedModel:
    model = LlamaBidirectionalModel(config)

    model.embed_tokens.weight.requires_grad = False
    model.embed_tokens.weight.zero_()

    return model