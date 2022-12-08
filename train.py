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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

encoded_image_size = 16
img_feature_channels = 2048 # Resnet 101 [:-2] layer output dim
embed_dim = 100
gradient_clipping = 2.0
learning_rate = 0.00008
max_len=22
nhead=6
num_decoder_layers=5
# dim_feedforward=512
dropout=0.1

# Training parameters
start_epoch = 0
epochs = 60  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
# decoder_lr = 4e-4  # learning rate for decoder
# decoder_lr = 0.0005
weight_decay = 0.5
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 10  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
ckpt_dir_prefix = f"ckpt_decoder_only_{nhead}_{num_decoder_layers}/"
if not os.path.exists(ckpt_dir_prefix):
   os.makedirs(ckpt_dir_prefix)
checkpoint = None 
# checkpoint = ckpt_dir_prefix + "new_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar"


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
        decoder = transformer.Decoder(len(word_map), img_feature_channels, embed_dim, num_decoder_layers, nhead, device=device)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, weight_decay=weight_decay)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        bleu1 = checkpoint['bleu-1']
        bleu2 = checkpoint['bleu-2']
        bleu3 = checkpoint['bleu-3']
        bleu4 = checkpoint['bleu-4']
        # best_bleu4 = checkpoint['bleu-4']
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

    print("------", "Training started for " + ckpt_dir_prefix + " with learning rate", learning_rate, "------")

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
        recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4 = validate(val_loader=val_loader,
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
        save_checkpoint_new(ckpt_dir_prefix, data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, 
                            recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4, is_best, 
                            learning_rate, None, embed_dim, nhead, None,
                            num_decoder_layers, None, dropout)


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

            rev_word_map = {v: k for k, v in word_map.items()} 
            captions = [[rev_word_map[int(ind)] for ind in cap if rev_word_map[int(ind)] not in ['<start>', '<end>', '<pad>']] for cap in caps]
            references.extend(captions)
            preds = greedy_decoding(decoder, encoder_out, 2631, 2632, 0, rev_word_map, 22, device)
            preds = [[char for char in pred if char not in ['<start>', '<end>', '<pad>']] for pred in preds]
            hypotheses.extend(preds)
            if i % 20 == 0:
                rand_idx = np.random.randint(0, len(captions))
                print("Ref", captions[rand_idx])
                print("Pred", preds[rand_idx])



        # Calculate BLEU-4 scores
        bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0))
        bleu4 = corpus_bleu(references, hypotheses)

        print('BLEU-1 - {bleu1}, BLEU-2 - {bleu2}, BLEU-3 - {bleu3}, BLEU-4 - {bleu4}\n'.format(bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4))

    return bleu1, bleu2, bleu3, bleu4


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
