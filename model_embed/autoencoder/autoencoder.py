import torch.nn as nn
import torch

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=1024
        )
        self.encoder_output_layer = nn.Linear(
            in_features=1024, out_features=kwargs["output_shape"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs["output_shape"], out_features=1024
        )
        self.decoder_output_layer = nn.Linear(
            in_features=1024, out_features=kwargs["input_shape"]
        )

    def forward(self, features,train=True):
        if train:
            activation = self.encoder_hidden_layer(features)
            activation = torch.relu(activation)
            code = self.encoder_output_layer(activation)
            code = torch.relu(code)
            activation = self.decoder_hidden_layer(code)
            activation = torch.relu(activation)
            activation = self.decoder_output_layer(activation)
            reconstructed = torch.relu(activation)
            return reconstructed
        else:
            activation = self.encoder_hidden_layer(features)
            activation = torch.relu(activation)
            code = self.encoder_output_layer(activation)
            code = torch.relu(code)
            return code

