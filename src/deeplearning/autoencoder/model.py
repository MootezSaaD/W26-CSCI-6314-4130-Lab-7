import torch.nn as nn
import torch

class TransformerAutoEncoder(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 bottleneck_dim: int, 
                 bert_encoder: nn.Module
                 ) -> None:
        super(TransformerAutoEncoder, self).__init__()

        self.bert_encoder = bert_encoder
        assert bottleneck_dim < hidden_dim, "Bottleneck dimension should be less than hidden dimension"
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedding = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        encoded = self.encoder(embedding)
        decoded = self.decoder(encoded)
        return embedding, decoded
    
    def get_bottleneck_representation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Direct method to get compressed representations"""
        with torch.no_grad():
            outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            return self.encoder(outputs)