import time
import os

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from default_transformer import ImageTransformer
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = 'flickr8k_output'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

# Model parameters
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
print("DEVICE:", device)
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

encoded_image_size = 8
learning_rate = 0.0001
patch_size=8
embed_dim=128
max_len=22
nhead=16
num_encoder_layers=6
num_decoder_layers=6
dim_feedforward=1024
dropout=0.5

# Training parameters
start_epoch = 0
epochs = 30  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 8
workers = 0  # for data-loading; right now, only 1 works with h5py
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches

ckpt_dir_prefix = f"ckpt_{nhead}_{num_decoder_layers}/"
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

    # Initialize / load checkpoint
    if checkpoint is None:
        model = ImageTransformer(len_vocab=len(word_map), img_size=256, patch_size=patch_size, in_chans=3, embed_dim=embed_dim, max_len=max_len, 
                                nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        model = model.to(device)
        optimizer = model.configure_optimizers(lr=learning_rate)
    else:
        checkpoint = torch.load(checkpoint)
        print("Last epoch", checkpoint['epoch'])
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        bleu1 = checkpoint['bleu-1']
        bleu2 = checkpoint['bleu-2']
        bleu3 = checkpoint['bleu-3']
        bleu4 = checkpoint['bleu-4']
        # best_bleu4 = checkpoint['bleu-4']
        model = checkpoint['model']
        model = model.to(device)
        optimizer = checkpoint['optimizer']


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
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              opt=optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4 = validate(val_loader=val_loader, model=model)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint_new(ckpt_dir_prefix, data_name, epoch, epochs_since_improvement, model, optimizer, 
                            recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4, is_best, 
                            learning_rate, patch_size, embed_dim, nhead, num_encoder_layers,
                            num_decoder_layers, dim_feedforward, dropout)
        # save_checkpoint_test(ckpt_dir_prefix, data_name, epoch, epochs_since_improvement, model, optimizer, 
                            # recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4, is_best, 
                            # learning_rate, patch_size, embed_dim, nhead, num_encoder_layers,
                            # num_decoder_layers, dim_feedforward, dropout)


def train(train_loader, model, opt, epoch):
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

    model.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    
    # Batches
    outs = []
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        if i == len(train_loader) - 1:
            break

        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        loss = model.training_step((imgs, caps), i)
        outs.append(loss.detach())

        opt.zero_grad()
        loss.backward()
        opt.step()
       
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})'.format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        top5=top5accs))
                                                                    

def validate(val_loader, model):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    model.eval()  # eval mode (no dropout or batchnorm)

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

            rev_word_map = {v: k for k, v in word_map.items()}

            captions = [[rev_word_map[int(ind)] for ind in cap if rev_word_map[int(ind)] not in ['<start>', '<end>', '<pad>']] for cap in caps]
            references.extend(captions)

            preds = model.predict(imgs)
            preds = [[rev_word_map[int(ind)] for ind in pred if rev_word_map[int(ind)] not in ['<start>', '<end>', '<pad>']] for pred in preds]
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


if __name__ == '__main__':
    main()
