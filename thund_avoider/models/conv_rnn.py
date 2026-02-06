import torch
import torch.nn as nn


class ConvRNNCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvRNNCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Ensures output has the same size
        self.bias = bias

        # Define the convolutional layer
        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, hidden_state):
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        hidden_state = torch.tanh(self.conv(combined))
        return hidden_state

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device,)


class ConvRNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, device):
        super(ConvRNNEncoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            self.cells.append(
                ConvRNNCell(
                    input_channels=cur_input_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                ).to(self.device)
            )

    def forward(self, input_tensor):
        batch_size, seq_len, _, height, width = input_tensor.size()
        hidden_state = self._init_hidden(batch_size, (height, width))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cells[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], hidden_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.cells[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]


class ConvRNNDecoder(nn.Module):
    def __init__(self, hidden_channels, output_channels, kernel_size, num_layers, device):
        super(ConvRNNDecoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_channels = hidden_channels if i == 0 else hidden_channels
            self.cells.append(
                ConvRNNCell(
                    input_channels=cur_input_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                ).to(self.device)
            )

        self.decoder = nn.Conv2d(hidden_channels, output_channels, kernel_size=1).to(device)
        self.adjust_channels = nn.Conv2d(output_channels, hidden_channels, kernel_size=3, padding=1).to(self.device)

    def forward(self, hidden_state, future_seq_len):
        h = hidden_state[-1]  # Start from the last hidden state of the encoder

        predicted_sequence = []
        for _ in range(future_seq_len):
            next_frame = self.decoder(h)
            next_frame_adjusted = self.adjust_channels(next_frame)
            predicted_sequence.append(next_frame)
            h = self.cells[-1](input_tensor=next_frame_adjusted, hidden_state=h)

        predicted_sequence = torch.stack(predicted_sequence, dim=1)
        return predicted_sequence


class ConvRNNSeq2Seq(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size, num_layers, device):
        super(ConvRNNSeq2Seq, self).__init__()
        self.encoder = ConvRNNEncoder(input_channels, hidden_channels, kernel_size, num_layers, device).to(device)
        self.decoder = ConvRNNDecoder(hidden_channels, output_channels, kernel_size, num_layers, device).to(device)

    def forward(self, input_sequence, future_seq_len):
        _, last_state_list = self.encoder(input_sequence)
        predicted_sequence = self.decoder(last_state_list, future_seq_len)
        return predicted_sequence
