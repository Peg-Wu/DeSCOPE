from transformers import PretrainedConfig


class DeSCOPEConfig(PretrainedConfig):
    model_type = "descope"
    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=672,
        hidden_dropout=0,
        pert_gene_encoder_layers=1,
        variational_encoder_layers=4,
        variational_decoder_layers=4,
        input_pert_gene_embedding_size=5120,  # ESM2: 5120
        input_length=18080,  # rna, atac
        add_layernorm=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.pert_gene_encoder_layers = pert_gene_encoder_layers
        self.variational_encoder_layers = variational_encoder_layers
        self.variational_decoder_layers = variational_decoder_layers
        self.input_pert_gene_embedding_size = input_pert_gene_embedding_size
        self.input_length = input_length
        self.add_layernorm = add_layernorm