import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, Transformer
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

import transformer

# Data parameters
data_folder = 'flickr8k_output'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
print("DEVICE:", device)
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

encoded_image_size = 8
img_feature_channels = 2048 # Resnet 101 [:-2] layer output dim
embedding_dim = 100

learning_rate = 0.0001
patch_size=16
embed_dim=128
max_len=22
nhead=8
num_encoder_layers=4
num_decoder_layers=4
dim_feedforward=1024
dropout=0.5

# Training parameters
start_epoch = 0
epochs = 60  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning

best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?

ckpt_dir_prefix = f"ckpt_rn_transformer_{nhead}_{num_decoder_layers}_1024_0.5/"
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
        decoder = Transformer(len_vocab=len(word_map), img_size=256, patch_size=patch_size, in_chans=3, embed_dim=embed_dim, max_len=max_len, 
                                nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                                dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        decoder = decoder.to(device)
        decoder_optimizer = decoder.configure_optimizers(lr=learning_rate)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        # best_bleu4 = checkpoint['bleu-4']
        bleu1 = checkpoint['bleu-1']
        bleu2 = checkpoint['bleu-2']
        bleu3 = checkpoint['bleu-3']
        bleu4 = checkpoint['bleu-4']
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
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4 = validate(val_loader=val_loader,
                                                                            encoder=encoder,
                                                                            decoder=decoder)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(ckpt_dir_prefix, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                            recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4, is_best, 
                            learning_rate, patch_size, embed_dim, nhead, num_encoder_layers,
                            num_decoder_layers, dim_feedforward, dropout)


def train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, epoch):
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

        loss = decoder.training_step((encoder_out, caps), i)

        decoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
                                                                    

def validate(val_loader, encoder, decoder):
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
            encoder_out = encoder(imgs)
            encoder_out = encoder_out.reshape(batch_size, -1, encoder_out.shape[-1]) 
            encoder_out = encoder_out.detach()
            preds = decoder.predict(encoder_out)

            rev_word_map = {v: k for k, v in word_map.items()} 
            captions = [[rev_word_map[int(ind)] for ind in cap if rev_word_map[int(ind)] not in ['<start>', '<end>', '<pad>']] for cap in caps]
            references.extend(captions)

            hyp = [[rev_word_map[int(ind)] for ind in pred if rev_word_map[int(ind)] not in ['<start>', '<end>', '<pad>']] for pred in preds]
            hypotheses.extend(hyp)

            start = time.time()

            if i % 20 == 0:
                rand_idx = np.random.randint(0, len(captions))
                print("Ref", captions[rand_idx])
                print("Pred", hyp[rand_idx])

        # Calculate BLEU-4 scores
        bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0))
        bleu4 = corpus_bleu(references, hypotheses)

        print('BLEU-1 - {bleu1}, BLEU-2 - {bleu2}, BLEU-3 - {bleu3}, BLEU-4 - {bleu4}\n'.format(bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4))

    return bleu1, bleu2, bleu3, bleu4


if __name__ == '__main__':
    main()
