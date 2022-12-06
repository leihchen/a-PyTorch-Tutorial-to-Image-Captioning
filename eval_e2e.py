import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
data_folder = 'flickr8k_output'
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
checkpoint = 'checkpoints/new_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size, max_length=15):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    # Batched Beam Search, thus do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for batch_idx, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        logits = model(image, torch.tensor([word_map['<start>']], dtype=torch.int32).to(device).reshape(1, -1))
        log_prob = torch.log(torch.nn.functional.softmax(logits.detach()[0][-1]))
        top_probs, top_indices = torch.topk(log_prob, k)
        targets = []
        prob_sums = []
        final = []
        final_prob_sums = []
        for i in range(k):
            token = top_indices[i]
            if token == word_map['<end>']:
                final.append(torch.tensor([word_map['<start>'], token]).to(device))
                final_prob_sums.append(top_probs[i])
            else:
                targets.append(torch.tensor([word_map['<start>'], token]).to(device))
                prob_sums.append(top_probs[i])
        k -= len(final)
        
        for itr in range(2, max_length):
            log_prob_beams = []
            log_prob_beams_norm = []
            for i in range(k):
                logits = model(image, torch.tensor(targets[i]).to(device).view(1, -1))
                log_prob = torch.log(torch.nn.functional.softmax(logits.detach()[0][-1]))
                log_prob_beams.append(log_prob)
                log_prob_norm = (log_prob + prob_sums[i]) / (len(targets[i]) + 1)
                log_prob_beams_norm.append(log_prob_norm)

            log_prob_beams = torch.cat(log_prob_beams)
            log_prob_beams_norm = torch.cat(log_prob_beams_norm, 0)
            top_probs, top_indices = torch.topk(log_prob_beams_norm, k)
            new_targets = []
            new_prob_sums = []
            for i in range(k):
                token = top_indices[i] % (vocab_size)
                prev_idx = top_indices[i] // (vocab_size)
                if token == word_map['<end>']:
                    final.append(torch.cat((targets[prev_idx], torch.tensor([token]).to(device))))
                    final_prob_sums.append(prob_sums[prev_idx] + log_prob_beams[top_indices[i]])
                else:
                    new_targets.append(torch.cat((targets[prev_idx], torch.tensor([token]).to(device))))
                    new_prob_sums.append(prob_sums[prev_idx] + log_prob_beams[top_indices[i]])
            k -= len(targets) - len(new_targets)
            targets = new_targets
            prob_sums = new_prob_sums
            if k == 0:
                break
            
        for i in range(k):
            final.append(targets[i])
            final_prob_sums.append(prob_sums[i])

        final_prob_sums = [prob.to(torch.device("cpu")) for prob in final_prob_sums]
        final_prob_avg = np.array(final_prob_sums) / np.array([len(final[i]) for i in range(len(final))])
        best_idx = np.argmax(final_prob_avg) 
        best_pred = final[best_idx]

        ref = [rev_word_map[int(ind)] for ind in caps[0] if rev_word_map[int(ind)] not in ['<start>', '<end>', '<pad>']]
        hyp = [rev_word_map[int(ind)] for ind in best_pred if rev_word_map[int(ind)] not in ['<start>', '<end>', '<pad>']]
        references.append(ref)
        hypotheses.append(hyp)

        if batch_idx % 10 == 0:
            print("Ref", ref)
            print("Pred", hyp)

        if batch_idx == 1000:
            break
        

    # Calculate BLEU-4 scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0))
    bleu4 = corpus_bleu(references, hypotheses)

    print('BLEU-1 - {bleu1:.4f}, BLEU-2 - {bleu2:.4f}, BLEU-3 - {bleu3:.4f}, BLEU-4 - {bleu4:.4f}\n'.format(bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4))

    return bleu1, bleu2, bleu3, bleu4


if __name__ == '__main__':
    beam_size = 2
    evaluate(beam_size)
