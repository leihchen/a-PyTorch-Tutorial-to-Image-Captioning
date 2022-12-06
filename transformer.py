
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss, LeakyReLU, Sequential
from torch.optim import Adam


class ResidualBlock(Module):
    """Represents 1D version of the residual block: https://arxiv.org/abs/1512.03385"""

    def __init__(self, input_dim):
        """Initializes the module."""
        super(ResidualBlock, self).__init__()
        self.block = Sequential(
            Linear(input_dim, input_dim),
            LeakyReLU(),
            Linear(input_dim, input_dim),
        )

    def forward(self, x):
        """Performs forward pass of the module."""
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x


class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int, device=torch.device("cpu")) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        T = X.shape[1]
        pos = torch.arange(T).reshape(-1, 1)
        div = torch.exp(torch.arange(0, self.embedding_dim, 2) / self.embedding_dim * -math.log(10000))
        pos_enc = torch.zeros(T, self.embedding_dim).to(self.device)
        pos_enc[:, 0::2] = torch.sin(pos * div)
        pos_enc[:, 1::2] = torch.cos(pos * div)
        return X + torch.broadcast_to(pos_enc.reshape(1, T, self.embedding_dim), (X.shape[0], T, self.embedding_dim))


class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        Q = self.linear_Q(query_X) # batch_size x Q_seq_length x out_dim
        K = self.linear_K(key_X) # batch_size x K_seq_length x out_dim
        V = self.linear_V(value_X) # batch_size x K_seq_length x out_dim
        weights = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.out_dim)
        if mask is not None:
            weights = weights.masked_fill(torch.logical_not(mask), float(-1e32))
        weights = self.softmax(weights) # batch_size x Q_seq_length x K_seq_length
        output = torch.bmm(weights, V) # batch_size x Q_seq_length x out_dim
        return output, weights

        
class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights


class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """  
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)


class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """

        padding_locations = torch.where(X == self.vocab_size, torch.zeros_like(X, dtype=torch.float64),
                                        torch.ones_like(X, dtype=torch.float64))
        padding_mask = torch.einsum("bi,bj->bij", (padding_locations, padding_locations))

        X = self.embedding_layer(X)
        X = self.position_encoding(X)
        for block in self.blocks:
            X = block(X, padding_mask)
        return X, padding_locations


class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """  
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)
        
        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights


class Decoder(Module):

    def __init__(self, vocab_size: int, img_feature_channels: int, embedding_dim: int, n_blocks: int, n_heads: int, device=torch.device("cpu")) -> None:
        super().__init__()
        
        self.entry_mapping_img = Linear(img_feature_channels, embedding_dim)
        self.res_block = ResidualBlock(embedding_dim)

        word_embeddings = torch.Tensor(np.loadtxt("embeddings.txt"))
        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=0
        )

        # self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim, device)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        # transformer_decoder_layer = torch.nn.TransformerDecoderLayer(
        #     d_model=embedding_dim,
        #     nhead=n_heads,
        #     dim_feedforward=1024,
        #     dropout=0.5
        # )
        # self.decoder = torch.nn.TransformerDecoder(transformer_decoder_layer, n_blocks)

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size
        self.device = device

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        return torch.tril(torch.ones(seq_length, seq_length))


    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        
        # Lookahead mask
        seq_length = target.shape[1]
        mask = self._lookahead_mask(seq_length).to(self.device)
        bs, source_seq_length, _ = encoded_source.shape
        encoded_source = self.entry_mapping_img(encoded_source)
        encoded_source = torch.nn.functional.leaky_relu(encoded_source)

        # Padding masks
        target_padding = torch.where(target == 0, torch.zeros_like(target, dtype=torch.float32), 
                                     torch.ones_like(target, dtype=torch.float32))
        target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, target_padding)).to(self.device)
        mask1 = torch.multiply(mask, target_padding_mask)

        if source_padding == None:
            source_padding = torch.ones((bs, source_seq_length), dtype=torch.float32).to(self.device)
        source_target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, source_padding))

        target = self.embedding_layer(target)
        target = self.position_encoding(target)

        att_weights = None
        for block in self.blocks:
            target, att = block(encoded_source, target, mask1, source_target_padding_mask)
            if att_weights is None:
                att_weights = att
        
        # target = self.decoder(
        #     tgt=target.permute(1, 0, 2),
        #     memory=encoded_source.permute(1, 0, 2),
        #     tgt_key_padding_mask=target_padding.bool(),
        #     tgt_mask=mask.bool()
        # )
        # y = self.linear(target.permute(1, 0, 2))

        y = self.linear(target)
        return y, att_weights
        # return y, _


class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)

    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def predict(self, source: List[int], beam_size=1, max_length=12) -> List[int]:
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.

        Hint: The follow functions may be useful:
            - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
            - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
        """
        self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)
        
        if not isinstance(source, torch.Tensor):
            source_input = torch.tensor(source).view(1, -1)
        else:
            source_input = source.view(1, -1)
        
        # TODO: Implement beam search.
        logits, _ = self.forward(source_input, torch.tensor([0], dtype=torch.int32).reshape(1, -1))
        log_prob = torch.log(torch.nn.functional.softmax(logits.detach()[0][-1]))
        top_probs, top_indices = torch.topk(log_prob, beam_size)
        targets = []
        prob_sums = []
        final = []
        final_prob_sums = []
        for i in range(beam_size):
            token = top_indices[i]
            if token == 1:
                final.append(torch.tensor([0, token]))
                final_prob_sums.append(top_probs[i])
            else:
                targets.append(torch.tensor([0, token]))
                prob_sums.append(top_probs[i])
        beam_size -= len(final)
        
        for itr in range(2, max_length):
            log_prob_beams = []
            log_prob_beams_norm = []
            for i in range(beam_size):
                logits, _ = self.forward(source_input, torch.tensor(targets[i]).view(1, -1))
                log_prob = torch.log(torch.nn.functional.softmax(logits.detach()[0][-1]))
                log_prob_beams.append(log_prob)
                log_prob_beams_norm.append((log_prob + prob_sums[i]) / (len(targets[i]) + 1))
            log_prob_beams = torch.cat(log_prob_beams)
            log_prob_beams_norm = torch.cat(log_prob_beams_norm)
            top_probs, top_indices = torch.topk(log_prob_beams_norm, beam_size)
            new_targets = []
            new_prob_sums = []
            for i in range(beam_size):
                token = top_indices[i] % (self.decoder.vocab_size + 1)
                prev_idx = top_indices[i] // (self.decoder.vocab_size + 1)
                if token == 1:
                    final.append(torch.cat((targets[prev_idx], torch.tensor([token]))))
                    final_prob_sums.append(prob_sums[prev_idx] + log_prob_beams[top_indices[i]])
                else:
                    new_targets.append(torch.cat((targets[prev_idx], torch.tensor([token]))))
                    new_prob_sums.append(prob_sums[prev_idx] + log_prob_beams[top_indices[i]])
            beam_size -= len(targets) - len(new_targets)
            targets = new_targets
            prob_sums = new_prob_sums
            if beam_size == 0:
                break
            
        for i in range(beam_size):
            final.append(targets[i])
            final_prob_sums.append(prob_sums[i])
        final_prob_avg = np.array(final_prob_sums) / np.array([len(final[i]) for i in range(len(final))])
        best_idx = np.argmax(final_prob_avg) 
        return final[best_idx], final_prob_avg[best_idx]


def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab


def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):
    
    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)


def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)


def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 1:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 1:
        target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    plt.show()
    plt.close()


def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")
    return epoch_train_loss, epoch_test_loss


def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    *** For students in 10-617 only ***
    (Students in 10-417, you can leave `raise NotImplementedError()`)

    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (0), EOS (1), and padding (anything after EOS)
    from the predicted and target sentences.
    
    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """
    if 1 in predicted:
        predicted = predicted[1:predicted.index(1)]
    else:
        predicted = predicted[1:]
    target = target[1:target.index(1)]

    if len(predicted) < N or len(target) < N:
        return 0

    weighted_geo_mean = 1
    for n in range(1, N + 1):
        n_grams_pred = {}
        for i in range(len(predicted) - n + 1):
            n_gram = tuple(predicted[i:i+n])
            if n_gram not in n_grams_pred:
                n_grams_pred[n_gram] = 0
            n_grams_pred[n_gram] += 1
        n_grams_targ = {}
        for i in range(len(target) - n + 1):
            n_gram = tuple(target[i:i+n])
            if n_gram in n_grams_pred:
                if n_gram not in n_grams_targ:
                    n_grams_targ[n_gram] = 0
                n_grams_targ[n_gram] += 1
        prob = 0
        for n_gram in n_grams_pred:
            targ_count = 0
            if n_gram in n_grams_targ:
                targ_count = n_grams_targ[n_gram]
            prob += min(n_grams_pred[n_gram], targ_count)
        prob = prob / (len(predicted) - n + 1)
        weighted_geo_mean *= prob**(1/N)
    
    return weighted_geo_mean * min(1, math.exp(1 - len(target) / len(predicted)))


if __name__ == "__main__":
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()
    
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), 12)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), 12)

    n_encoder_blocks = 2
    n_decoder_blocks = 4
    n_heads = 3
    model = Transformer(len(source_vocab), len(target_vocab), 256, n_encoder_blocks, n_decoder_blocks, n_heads)
    
    # Q2
    # epoch_train_loss, epoch_test_loss = train(model, train_source, train_target, test_source, test_target, len(target_vocab),)

    # torch.save(model.state_dict(), f'model_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}.pkl')

    # plt.plot(np.arange(1, len(epoch_train_loss) + 1), epoch_train_loss, label="Train Loss")
    # plt.plot(np.arange(1, len(epoch_train_loss) + 1), epoch_test_loss, label="Test Loss")
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title(f"Train and Test Loss for {n_encoder_blocks} encoder blocks, {n_decoder_blocks} decoder blocks, and {n_heads} attention heads")
    # plt.show()

    #Q3
    # model.load_state_dict(torch.load(f'model_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}.pkl'))
    # for i in range(8):
    #     source, target = test_source[i], test_target[i]
    #     pred, likelihood = model.predict(source, beam_size=3)
    #     print(decode_sentence(source, source_vocab))
    #     print(decode_sentence(target, target_vocab))
    #     print(decode_sentence(pred, target_vocab))
    #     print(likelihood)
    #     print()

    #Q4
    # model.load_state_dict(torch.load(f'model_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}.pkl'))
    # for i in range(3):
    #     source, target = train_source[i], train_target[i]
    #     _, att_weight = model(source.view(1, -1), target.view(1, -1))
    #     print(att_weight.shape)
    #     for j in range(n_heads):
    #         visualize_attention(source.detach().numpy(), target.detach().numpy(), source_vocab, target_vocab, att_weight[0][j].detach().numpy())

    #Q5
    # model.load_state_dict(torch.load(f'model_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}.pkl'))
    # avg = []
    # for beam_size in range(1, 9):
    #     likelihood_sum = 0
    #     for i in range(100):
    #         source, target = test_source[i], test_target[i]
    #         pred, likelihood = model.predict(source, beam_size=beam_size)
    #         likelihood_sum += likelihood
    #     avg.append(likelihood_sum / 100)
    # plt.plot(np.arange(1, 9), avg)
    # plt.xlabel('Beam Size')
    # plt.ylabel('Avg Norm Log-likelihood')
    # plt.legend()
    # plt.title(f"Average Normalized Log-Likelihood for Different Beam Size")
    # plt.show()
    
    #Q6
    # n_encoder_blocks = 2
    # n_decoder_blocks = 2
    # n_heads = 3
    # model = Transformer(len(source_vocab), len(target_vocab), 256, n_encoder_blocks, n_decoder_blocks, n_heads)
    # model.load_state_dict(torch.load(f'model_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}.pkl'))
    # for k in range(1, 5):
    #     bleu_sum = 0
    #     for i in tqdm(range(len(test_source))):
    #         source, target = test_source[i], test_target[i]
    #         pred, likelihood = model.predict(source)
    #         pred, target = list(pred.detach().numpy()), list(target.detach().numpy())
    #         score = bleu_score(pred, target, k)
    #         bleu_sum += score
    #     print(f"Average Bleu-{k} score: {round(bleu_sum / len(test_source), 4)}")
