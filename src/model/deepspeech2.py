import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpeech2(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_tokens, num_rnn_layers, bidirectional):
        """
        DeepSpeech2 model with GRU and two convolutional layers.

        Args:
            input_dim (int): Dimension of the input feature (height of the spectrogram frequencies).
            hidden_dim (int): Number of neurons in the RNN hidden layer.
            n_tokens (int): Number of vocab.
            num_rnn_layers (int): Number of RNN layers.
            bidirectional (bool): If True, RNN will be bidirectional.
        """
        super(DeepSpeech2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(11, 41),
                stride=(2, 2),
                padding=(5, 20)
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(11, 21),
                stride=(1, 2),
                padding=(5, 10)
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        self.gru_input_size = self._calc_rnn_inp_dim(input_dim)

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, n_tokens
        )

    def _conv_output_size(self, input_size, kernel_size, stride, padding):
            return (input_size + 2 * padding - kernel_size) // stride + 1

    def _calc_rnn_inp_dim(self, input_dim):
        freq_after_conv1 = self._conv_output_size(
            input_dim, kernel_size=11, stride=2, padding=5
        )
        freq_after_conv2 = self._conv_output_size(
            freq_after_conv1, kernel_size=11, stride=1, padding=5
        )

        return 32 * freq_after_conv2  # 32 channels * freq_after_conv2

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass for the DeepSpeech2 model with GRU and two convolutional layers.

        Args:
            spectrogram (Tensor):mel-spectrogram with shape (B, freq, time).
            spectrogram_length (Tensor): lengths of each spectrogram in the batch.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """

        # print("========spectrogram_shape", spectrogram.shape)
        batch_size = spectrogram.size(0)
        x = spectrogram.unsqueeze(1)        # (B, 1, freq, time)

        x = self.conv(x)  # (B, C, freq, time)

        spectrogram_length = self._conv_output_size(
            spectrogram_length, kernel_size=41, stride=2, padding=20
        )
        spectrogram_length = self._conv_output_size(
            spectrogram_length, kernel_size=21, stride=2, padding=10
        )

        # print("========a_conv_x_shape", x.shape)
        x = x.permute(0, 3, 1, 2)                           # (B, time, C, freq)
        x = x.contiguous().view(batch_size, x.size(1), -1)  # (B, time, C * freq)

        # print("========a_rnn_x_shape", x.shape)
        x = nn.utils.rnn.pack_padded_sequence(x, spectrogram_length.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # Shape: (batch, time, hidden_size * num_directions)
        x = self.fc(x)  # (batch, time, n_tokens)

        log_probs = F.log_softmax(x, dim=-1)

        output = {
            "log_probs": log_probs,
            "log_probs_length": spectrogram_length
        }

        return output

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
