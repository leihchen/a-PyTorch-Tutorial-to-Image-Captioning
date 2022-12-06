import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

import transformer

# Data parameters
# data_folder = 'flickr30k_output'  # folder with data files saved by create_input_files.py
data_folder = 'flickr8k_output'
# data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

encoded_image_size = 8
img_feature_channels = 2048 # Resnet 101 [:-2] layer output dim
embedding_dim = 100
num_decoder_blocks = 3
num_decoder_heads = 5
gradient_clipping = 2.0

# Training parameters
start_epoch = 0
epochs = 60  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
# decoder_lr = 4e-4  # learning rate for decoder
decoder_lr = 0.00001
weight_decay = 0.5
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    print(len(word_map))

    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = Encoder(encoded_image_size=encoded_image_size)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        # decoder = DecoderWithAttention(attention_dim=attention_dim,
        #                                embed_dim=emb_dim,
        #                                decoder_dim=decoder_dim,
        #                                vocab_size=len(word_map),
        #                                dropout=dropout)
        # decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
        #                                      lr=decoder_lr)
        decoder = transformer.Decoder(len(word_map), img_feature_channels, embedding_dim, num_decoder_blocks, num_decoder_heads, device=device)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True) 

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        if i == len(train_loader) - 1:
            break

        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        with torch.no_grad():
            encoder_out = encoder(imgs)
            encoder_out = encoder_out.reshape(batch_size, -1, encoder_out.shape[-1]) 
            encoder_out = encoder_out.detach()

        pred, alphas = decoder(encoder_out, None, caps)
        pred = pred[:, :-1, :]
        caps = caps[:, 1:]

        tgt_padding_mask = torch.where(caps == 0, torch.zeros_like(caps, dtype=torch.float32), torch.ones_like(caps, dtype=torch.float32)).bool()

        # Calculate loss
        loss = criterion(pred[tgt_padding_mask], caps[tgt_padding_mask])
        
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clipping)

        # Clip gradients
        # if grad_clip is not None:
        #     clip_gradient(decoder_optimizer, grad_clip)
        #     if encoder_optimizer is not None:
        #         clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        decode_lengths = (caplens.squeeze(1) - 1).tolist()
        scores = pack_padded_sequence(pred, decode_lengths, batch_first=True, enforce_sorted=False).data
        targets = pack_padded_sequence(caps, decode_lengths, batch_first=True, enforce_sorted=False).data
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
                                                                    

def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            if i == len(val_loader) - 1:
                break
            
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                encoder_out = encoder(imgs)
            encoder_out = encoder_out.reshape(batch_size, -1, encoder_out.shape[-1])
            pred, alphas = decoder(encoder_out, None, caps)
            pred = pred[:, :-1, :]
            caps = caps[:, 1:]

            tgt_padding_mask = torch.where(caps == 0, torch.zeros_like(caps, dtype=torch.float32), torch.ones_like(caps, dtype=torch.float32)).bool()
            # Calculate loss
            loss = criterion(pred[tgt_padding_mask], caps[tgt_padding_mask])

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            # targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = pred.clone()
            decode_lengths = (caplens.squeeze(1) - 1).tolist()
            scores = pack_padded_sequence(scores_copy, decode_lengths, batch_first=True, enforce_sorted=False)
            targets = pack_padded_sequence(caps, decode_lengths, batch_first=True, enforce_sorted=False)

            # Add doubly stochastic attention regularization
            # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            # for j in range(allcaps.shape[0]):
            #     img_caps = allcaps[j].tolist()
            #     img_captions = list(
            #         map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
            #             img_caps))  # remove <start> and pads
            #     references.append(img_captions)
            
            # # Hypotheses
            # _, preds = torch.max(scores_copy, dim=2)
            # preds = preds.tolist()
            # temp_preds = list()
            # for j, p in enumerate(preds):
            #     temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            # preds = temp_preds
            # hypotheses.extend(preds)

            # assert len(references) == len(hypotheses)


            rev_word_map = {v: k for k, v in word_map.items()} 
            # for i in range(len(references)):
            #     ref, hyp = references[i], hypotheses[i]
            #     for i in range(len(ref)):
            #         ref_words = [rev_word_map[ind] for ind in ref[i]]
            #         print("REF", ref_words)
                
            #     hyp_words = [rev_word_map[ind] for ind in pred]
            #     print("Hyp", hyp_words)

            # for i in range(encoder_out.shape[0]):
            #     pred, _ = predict(decoder, encoder_out[i].reshape(1, encoder_out.shape[1], encoder_out.shape[2]), 2, 22)
            #     hypotheses.append(pred)
            #     if i == 0:
            #         ref_words = [rev_word_map[int(ind)] for ind in caps[0]]
            #         hyp_words = [rev_word_map[int(ind)] for ind in pred]
            #         print("REF", ref_words)
            #         print("HYP", hyp_words)
            captions = [[rev_word_map[int(ind)] for ind in cap] for cap in caps]
            references.extend(captions)
            pred = greedy_decoding(decoder, encoder_out, 2631, 2632, 0, rev_word_map, 22, device)
            hypotheses.extend(pred)
            if i == 0:
                print("Ref 0", captions[0])
                print("Pred 0", pred[0])
                print("Ref 1", captions[1])
                print("Pred 1", pred[1])
                print("Ref 2", captions[2])
                print("Pred 2", pred[2])


        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


def predict(decoder, source_input, beam_size=1, max_length=12):
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.
        """
        decoder.eval()

        logits, _ = decoder(source_input, None, torch.tensor([2631], dtype=torch.int32).to(device).reshape(1, -1))
        log_prob = torch.log(torch.nn.functional.softmax(logits.detach()[0][-1]))
        top_probs, top_indices = torch.topk(log_prob, beam_size)
        targets = []
        prob_sums = []
        final = []
        final_prob_sums = []
        for i in range(beam_size):
            token = top_indices[i]
            if token == 2632: # end token
                final.append(torch.tensor([2631, token]))
                final_prob_sums.append(top_probs[i])
            else:
                targets.append(torch.tensor([2631, token]))
                prob_sums.append(top_probs[i])
        beam_size -= len(final)
        
        for itr in range(2, max_length):
            log_prob_beams = []
            log_prob_beams_norm = []
            for i in range(beam_size):
                logits, _ = decoder(source_input, None, torch.tensor(targets[i]).to(device).view(1, -1))
                log_prob = torch.log(torch.nn.functional.softmax(logits.detach()[0][-1]))
                log_prob_beams.append(log_prob)
                log_prob_beams_norm.append((log_prob + prob_sums[i]) / (len(targets[i]) + 1))
            log_prob_beams = torch.cat(log_prob_beams)
            log_prob_beams_norm = torch.cat(log_prob_beams_norm)
            top_probs, top_indices = torch.topk(log_prob_beams_norm, beam_size)
            new_targets = []
            new_prob_sums = []
            for i in range(beam_size):
                token = top_indices[i] % (decoder.vocab_size + 1)
                prev_idx = top_indices[i] // (decoder.vocab_size + 1)
                if token == 2632: # end token
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
        final_prob_sums = [prob.to(torch.device("cpu")) for prob in final_prob_sums]
        final_prob_avg = np.array(final_prob_sums) / np.array([len(final[i]) for i in range(len(final))])
        best_idx = np.argmax(final_prob_avg) 
        return final[best_idx], final_prob_avg[best_idx]


def greedy_decoding(model, img_features_batched, sos_id, eos_id, pad_id, idx2word, max_len, device):
    """Performs greedy decoding for the caption generation.
    At each iteration model predicts the next word in the caption given the previously
    generated words and image features. For the next word we always take the most probable one.
    Arguments:
        model (torch.nn.Module): Transformer Decoder model which generates prediction for the next word
        img_features_padded (torch.Tensor): Image features generated by CNN encoder
            Stacked along 0-th dimension for each image in the mini-batch
        sos_id (int): Id of <start> token in the vocabulary
        eos_id (int): Id of <end> token in the vocabulary
        pad_id (int): Id of <pad> token in the vocabulary
        idx2word (dict): Mapping from ordinal number of token (i.e. class number) to the string of word
        max_len (int): Maximum length of the caption
        device (torch.device): Device on which to port used tensors
    Returns:
        generated_captions (list of str): Captions generated for each image in the batch
    """
    batch_size = img_features_batched.size(0)

    # Define the initial state of decoder input
    x_words = torch.Tensor([sos_id] + [pad_id] * (max_len - 1)).to(device).long()
    x_words = x_words.repeat(batch_size, 1)
    padd_mask = torch.Tensor([True] * max_len).to(device).bool()
    padd_mask = padd_mask.repeat(batch_size, 1)

    # Is each image from the batch decoded
    is_decoded = [False] * batch_size
    generated_captions = []
    for _ in range(batch_size):
        generated_captions.append([])

    for i in range(max_len - 1):
        # Update the padding masks
        padd_mask[:, i] = False

        # Get the model prediction for the next word
        y_pred_prob, _ = model(img_features_batched, None, x_words)
        # Extract the prediction from the specific (next word) position of the target sequence
        y_pred_prob = y_pred_prob[torch.arange(batch_size), [i] * batch_size].clone()
        # Extract the most probable word
        y_pred = y_pred_prob.argmax(-1)
        
        for batch_idx in range(batch_size):
            if is_decoded[batch_idx]:
                continue
            # Add the generated word to the caption
            generated_captions[batch_idx].append(idx2word[y_pred[batch_idx].item()])
            if y_pred[batch_idx] == eos_id:
                # Caption has been fully generated for this image
                is_decoded[batch_idx] = True
            
        if np.all(is_decoded):
            break

        if i < (max_len - 1):   # We haven't reached maximum number of decoding steps
            # Update the input tokens for the next iteration
            x_words[torch.arange(batch_size), [i+1] * batch_size] = y_pred.view(-1)

    # Complete the caption for images which haven't been fully decoded
    for batch_idx in range(batch_size):
        if not is_decoded[batch_idx]:
            generated_captions[batch_idx].append(idx2word[eos_id])

    # Clean the EOS symbol
    for caption in generated_captions:
        caption.remove("<end>")

    return generated_captions


if __name__ == '__main__':
    main()
