import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # [num_layers, b, hidden_size]
        out, hn = self.rnn(x, h0)  # [b, t, hidden_size]
        return out, hn  # Return the final hidden state


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return out, (hn, cn)  # Return output and final hidden & cell states


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)
        return out, hn  # Return output and final hidden state


class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_to_hidden = nn.Linear(output_size, hidden_size)

    def forward(self, x, hn):
        out, hn = self.rnn(x, hn)  # Use the hidden state from the encoder as the initial hidden state for the decoder
        out = self.fc(out)  # Map to output space
        x = self.output_to_hidden(out)
        return out, hn, x


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_to_hidden = nn.Linear(output_size, hidden_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        x = self.output_to_hidden(out)  # Transform output to match the hidden_size
        return out, hidden, x


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_to_hidden = nn.Linear(output_size, hidden_size)

    def forward(self, x, hn):
        out, hn = self.gru(x, hn)
        out = self.fc(out)
        x = self.output_to_hidden(out)  # Transform output to match the hidden_size
        return out, hn, x


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, target_seq_length):
        encoder_output, encoder_hidden = self.encoder(x)  # Encode the input sequence
        decoder_input = encoder_output[:, -1:, :]  # Use the last hidden state of encoder as the first input for decoder
        decoder_hidden = encoder_hidden

        outputs = []
        for _ in range(target_seq_length):
            decoder_output, decoder_hidden, decoder_input  = self.decoder(decoder_input, decoder_hidden)
            outputs.append(decoder_output)

        outputs = torch.cat(outputs, dim=1)  # Concatenate outputs along sequence dimension
        return outputs
