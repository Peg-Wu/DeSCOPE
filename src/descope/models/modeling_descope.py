import torch
from torch import nn
from typing import Optional
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .configuration_descope import DeSCOPEConfig
from .utils import (
    build_mlp, 
    kl_loss_func,
    DeSCOPEForATACOutput,
    DeSCOPEForRNAOutput
)


class PertGeneEncoder(nn.Module):
    def __init__(self, config: DeSCOPEConfig):
        super().__init__()
        self.config = config
        self.n_encoder_layers = config.pert_gene_encoder_layers
        self.pert_gene_encoder = build_mlp(
            in_dim=config.input_pert_gene_embedding_size,
            out_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            n_layers=self.n_encoder_layers,
            dropout=config.hidden_dropout,
            activation=ACT2FN[config.hidden_act],
            add_layernorm=config.add_layernorm
        )
    
    def forward(self, x):
        # Input: (batch_size, input_pert_gene_embedding_size)
        # Output: (batch_size, hidden_size)
        return self.pert_gene_encoder(x)


class VariationalEncoder(nn.Module):
    def __init__(
        self, 
        config: DeSCOPEConfig,
        in_dim: int
    ):
        super().__init__()
        self.config = config
        self.n_encoder_layers = config.variational_encoder_layers
        self.encoder = build_mlp(
            in_dim=in_dim,
            out_dim=config.hidden_size * 2,  # mean, logvar
            hidden_dim=config.hidden_size,
            n_layers=self.n_encoder_layers,
            dropout=config.hidden_dropout,
            activation=ACT2FN[config.hidden_act],
            add_layernorm=config.add_layernorm
        )
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        # Input: (batch_size, in_dim)  # `in_dim = input_length + hidden_size` in `DeSCOPE`.
        # Output: 
        #   - z: (batch_size, hidden_size)
        #   - mean: (batch_size, hidden_size)
        #   - logvar: (batch_size, hidden_size)

        mean, logvar = torch.chunk(
            self.encoder(x),
            chunks=2, dim=-1
        )

        z = self.reparameterize(mean, logvar)
        return z, mean, logvar


class VariationalDecoder(nn.Module):
    def __init__(
        self, 
        config: DeSCOPEConfig,
        in_dim: int
    ):
        super().__init__()
        self.config = config
        self.n_decoder_layers = config.variational_decoder_layers
        self.decoder = build_mlp(
            in_dim=in_dim,
            out_dim=config.input_length,
            hidden_dim=config.hidden_size,
            n_layers=self.n_decoder_layers,
            dropout=config.hidden_dropout,
            activation=ACT2FN[config.hidden_act],
            add_layernorm=config.add_layernorm
        )

    def forward(self, x):
        # Input: (batch_size, in_dim)  # `indim = hidden_size * 2` in `DeSCOPE`.
        # Output: (batch_size, input_length)
        return self.decoder(x)


class DeSCOPEPretrainedModel(PreTrainedModel):
    config_class = DeSCOPEConfig

    # NOTE: Using pytorch default initialization.
    # def _init_weights(self, module):
    #     # std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         # module.weight.data.normal_(mean=0.0, std=std)
    #         # if module.bias is not None:
    #         #     module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         # module.weight.data.normal_(mean=0.0, std=std)
    #         # if module.padding_idx is not None:
    #         #     module.weight.data[module.padding_idx].zero_()


# NOTE: The RNA and ATAC models have identical architectures; 
# we maintain them as two distinct models to support independent adjustment for each data modality.
class DeSCOPEForATAC(DeSCOPEPretrainedModel):
    def __init__(
        self, 
        config: DeSCOPEConfig,
        alpha_mse_loss: float = 1.0,
        alpha_kl_loss: float = 1.0
    ):
        super().__init__(config)
        self.pert_gene_encoder = PertGeneEncoder(config)
        self.prior_encoder = VariationalEncoder(config, in_dim=config.input_length + config.hidden_size)
        self.posterior_encoder = VariationalEncoder(config, in_dim=config.input_length + config.hidden_size)
        self.decoder = VariationalDecoder(config, in_dim=config.hidden_size * 2)

        self.alpha_mse_loss = alpha_mse_loss
        self.alpha_kl_loss = alpha_kl_loss

        # Initialize weights and apply final processing.
        self.post_init()

    def forward(
        self, 
        ctrl_cell_tf_idf: Optional[torch.Tensor] = None,
        pert_gene_emb: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> DeSCOPEForATACOutput:
        pert_embeddings = self.pert_gene_encoder(pert_gene_emb)
        _, prior_mean, prior_logvar = self.prior_encoder(
            torch.cat([ctrl_cell_tf_idf, pert_embeddings], dim=-1)
        )
        posterior_z, posterior_mean, posterior_logvar = self.posterior_encoder(
            torch.cat([labels, pert_embeddings], dim=-1)
        )
        preds = self.decoder(torch.cat([posterior_z, pert_embeddings], dim=-1))
        preds = nn.functional.softplus(preds)

        # Loss
        mse_loss = nn.functional.mse_loss(preds, labels)
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            kl_loss = kl_loss_func(
                prior_mean.float(),
                prior_logvar.float(),
                posterior_mean.float(),
                posterior_logvar.float()
            )
        loss = self.alpha_mse_loss * mse_loss + self.alpha_kl_loss * kl_loss

        return DeSCOPEForATACOutput(
            loss=loss,
            mse_loss=mse_loss,
            kl_loss=kl_loss,
            logits=preds
        )
    
    @torch.no_grad()
    def inference(
        self,
        ctrl_cell_tf_idf: Optional[torch.Tensor] = None,
        pert_gene_emb: Optional[torch.Tensor] = None
    ) -> DeSCOPEForATACOutput:
        pert_embeddings = self.pert_gene_encoder(pert_gene_emb)
        prior_z, _, _ = self.prior_encoder(
            torch.cat([ctrl_cell_tf_idf, pert_embeddings], dim=-1)
        )
        preds = self.decoder(torch.cat([prior_z, pert_embeddings], dim=-1))
        preds = nn.functional.softplus(preds)

        return DeSCOPEForATACOutput(
            loss=None,
            mse_loss=None,
            kl_loss=None,
            logits=preds
        )


# NOTE: The RNA and ATAC models have identical architectures; 
# we maintain them as two distinct models to support independent adjustment for each data modality.
class DeSCOPEForRNA(DeSCOPEPretrainedModel):
    def __init__(
        self, 
        config: DeSCOPEConfig,
        alpha_mse_loss: float = 1.0,
        alpha_kl_loss: float = 1.0
    ):
        super().__init__(config)
        # self.pert_gene_encoder = PertGeneEncoder(config)
        self.pert_gene_encoder = nn.Sequential(
            nn.Linear(config.input_pert_gene_embedding_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        self.prior_encoder = VariationalEncoder(config, in_dim=config.input_length + config.hidden_size)
        self.posterior_encoder = VariationalEncoder(config, in_dim=config.input_length + config.hidden_size)
        self.decoder = VariationalDecoder(config, in_dim=config.hidden_size * 2)

        self.alpha_mse_loss = alpha_mse_loss
        self.alpha_kl_loss = alpha_kl_loss

        # Initialize weights and apply final processing.
        self.post_init()

    def forward(
        self, 
        ctrl_cell_expr: Optional[torch.Tensor] = None,
        pert_gene_emb: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> DeSCOPEForRNAOutput:
        pert_embeddings = self.pert_gene_encoder(pert_gene_emb)
        _, prior_mean, prior_logvar = self.prior_encoder(
            torch.cat([ctrl_cell_expr, pert_embeddings], dim=-1)
        )
        posterior_z, posterior_mean, posterior_logvar = self.posterior_encoder(
            torch.cat([labels, pert_embeddings], dim=-1)
        )
        preds = self.decoder(torch.cat([posterior_z, pert_embeddings], dim=-1))
        preds = nn.functional.softplus(preds)

        # Loss
        mse_loss = nn.functional.mse_loss(preds, labels)
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            kl_loss = kl_loss_func(
                prior_mean.float(),
                prior_logvar.float(),
                posterior_mean.float(),
                posterior_logvar.float()
            )
        loss = self.alpha_mse_loss * mse_loss + self.alpha_kl_loss * kl_loss

        return DeSCOPEForRNAOutput(
            loss=loss,
            mse_loss=mse_loss,
            kl_loss=kl_loss,
            logits=preds
        )
    
    @torch.no_grad()
    def inference(
        self,
        ctrl_cell_expr: Optional[torch.Tensor] = None,
        pert_gene_emb: Optional[torch.Tensor] = None
    ) -> DeSCOPEForRNAOutput:
        pert_embeddings = self.pert_gene_encoder(pert_gene_emb)
        prior_z, _, _ = self.prior_encoder(
            torch.cat([ctrl_cell_expr, pert_embeddings], dim=-1)
        )
        preds = self.decoder(torch.cat([prior_z, pert_embeddings], dim=-1))
        preds = nn.functional.softplus(preds)

        return DeSCOPEForRNAOutput(
            loss=None,
            mse_loss=None,
            kl_loss=None,
            logits=preds
        )