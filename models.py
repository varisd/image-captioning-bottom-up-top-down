import torch
import torchvision

from torch import nn
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x) + x


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self,
                 features_dim,
                 decoder_dim,
                 attention_dim,
                 dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()

        # linear layer to transform encoded image
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))

        # linear layer to transform decoder's output
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))

        # linear layer to calculate values to be softmax-ed
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # softmax layer to calculate weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension
            (batch_size, num_boxes, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension
            (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # (batch_size, 36, attention_dim)
        att1 = self.features_att(image_features)

        # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)

        # (batch_size, 36)
        att = self.full_att(
            self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)

        # (batch_size, 36) 
        alpha = self.softmax(att)

        # (batch_size, features_dim)
        attention_weighted_encoding = (
            image_features * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self,
                 attention_dim,
                 embed_dim,
                 decoder_dim,
                 vocab_size,
                 input_projection=None,
                 features_dim=2048,
                 dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # input projection
        self.input_projection = None
        if input_projection == "linear":
            self.input_projection = nn.Linear(features_dim, features_dim)
            self.input_projection.weight.data.uniform_(-0.1, 0.1)
            self.input_projection.bias.data.fill_(0)
        elif input_projection == "linear-id":
            self.input_projection = nn.Linear(features_dim, features_dim)
            self.input_projection.weight.data.copy_(torch.eye(features_dim))
            self.input_projection.bias.data.fill_(0)
        elif input_projection == "linear-res":
            proj = nn.Linear(features_dim, features_dim)
            proj.weight.data.uniform_(-0.1, 0.1)
            proj.bias.data.fill_(0)
            self.input_projection = Residual(proj)
        elif input_projection == "constant":
            self.input_projection = nn.Linear(features_dim, features_dim)
            self.input_projection.weight.data.copy_(torch.eye(features_dim))
            self.input_projection.bias.data.fill_(0)
            self.input_projection.weight.requires_grad = False
            self.input_projection.bias.requires_grad = False

        # attention network
        self.attention = Attention(features_dim, decoder_dim, attention_dim)

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dropout = nn.Dropout(p=self.dropout)

        # top down attention LSTMCell
        self.top_down_attention = nn.LSTMCell(
            embed_dim + features_dim + decoder_dim, decoder_dim, bias=True)

        # language model LSTMCell
        self.language_model = nn.LSTMCell(
            features_dim + decoder_dim, decoder_dim, bias=True)

        # linear layer to find scores over vocabulary
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))

        # initialize some layers with the uniform distribution
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution,
            for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self,batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based
            on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension
            (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # (batch_size, decoder_dim)
        h = torch.zeros(batch_size,self.decoder_dim).to(device)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension
            (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension
            (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension
            (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths,
            weights, sort indices
        """

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Input projection
        if self.input_projection is not None:
            image_features = self.input_projection(image_features)

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        image_features_mean = image_features.mean(1).to(device)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        # (batch_size, decoder_dim)
        h1, c1 = self.init_hidden_state(batch_size)
        h2, c2 = self.init_hidden_state(batch_size)
        
        # We won't decode at the <end> position, since we've finished
        # generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(
            batch_size, max(decode_lengths), vocab_size).to(device)
        predictions1 = torch.zeros(
            batch_size, max(decode_lengths), vocab_size).to(device)
        
        # At each time-step, pass the language model's previous hidden state,
        # the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass
        # the hidden state of the top down model and the bottom up 
        # features to the attention block. The attention weighed bottom up
        # features and hidden state of the top down attention model
        # are then passed to the language model 
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h1, c1 = self.top_down_attention(
                torch.cat(
                    [
                        h2[:batch_size_t],
                        image_features_mean[:batch_size_t],
                        embeddings[:batch_size_t, t, :]
                    ],
                    dim=1),
                (h1[:batch_size_t], c1[:batch_size_t]))
            attention_weighted_encoding = self.attention(
                image_features[:batch_size_t], h1[:batch_size_t])
            preds1 = self.fc1(self.dropout(h1))
            h2, c2 = self.language_model(
                torch.cat(
                    [
                        attention_weighted_encoding[:batch_size_t],
                        h1[:batch_size_t]
                    ],
                    dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))

            # (batch_size_t, vocab_size)
            preds = self.fc(self.dropout(h2))
            predictions[:batch_size_t, t, :] = preds
            predictions1[:batch_size_t, t, :] = preds1

        return (predictions, predictions1, encoded_captions, decode_lengths,
                sort_ind, image_features)
