import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Ensures output has the same size
        self.bias = bias

        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=2 * self.hidden_channels,  # Number of gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=self.hidden_channels,  # Number of gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, hidden_state):
        h_cur = hidden_state

        combined_1 = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis
        combined_conv = self.conv1(combined_1)
        z, r = torch.split(combined_conv, self.hidden_channels, dim=1)

        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined_2 = torch.cat([input_tensor, r * h_cur], dim=1)
        h_tilde = self.conv2(combined_2)
        h_tilde = torch.tanh(h_tilde)

        h_next = (1 - z) * h_cur + z * h_tilde
        return h_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv1.weight.device)


class ConvGRUEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, device):
        super(ConvGRUEncoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels
            self.cells.append(
                ConvGRUCell(
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
                h = self.cells[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], hidden_state=h
                    )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.cells[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]


class ConvGRUDecoder(nn.Module):
    def __init__(self, hidden_channels, output_channels, kernel_size, num_layers, device):
        super(ConvGRUDecoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_channels = hidden_channels if i == 0 else hidden_channels
            self.cells.append(
                ConvGRUCell(
                    input_channels=cur_input_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.kernel_size
                ).to(self.device)
            )

        self.decoder = nn.Conv2d(hidden_channels, output_channels, kernel_size=1, padding=0).to(self.device)
        self.adjust_channels = nn.Conv2d(output_channels, hidden_channels, kernel_size=3, padding=1).to(self.device)

    def forward(self, last_state_list, future_seq_len):
        h = last_state_list[-1]  # Use the hidden state from the last encoder layer

        predicted_sequence = []
        for _ in range(future_seq_len):
            next_frame = self.decoder(h)
            next_frame_adjusted = self.adjust_channels(next_frame)
            predicted_sequence.append(next_frame)
            h = self.cells[-1](input_tensor=next_frame_adjusted, hidden_state=h)

        predicted_sequence = torch.stack(predicted_sequence, dim=1)
        return predicted_sequence


class ConvGRUSeq2Seq(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size, num_layers, device):
        super(ConvGRUSeq2Seq, self).__init__()
        self.encoder = ConvGRUEncoder(input_channels, hidden_channels, kernel_size, num_layers, device).to(device)
        self.decoder = ConvGRUDecoder(hidden_channels, output_channels, kernel_size, num_layers, device).to(device)

    def forward(self, input_sequence, future_seq_len):
        _, last_state_list = self.encoder(input_sequence)
        predicted_sequence = self.decoder(last_state_list, future_seq_len)
        return predicted_sequence
