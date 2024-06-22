"""
Author: Prakhar
"""
import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def choose_from_top_k_top_n(probs, k=50, p=0.8):
    ind = np.argpartition(probs, -k)[-k:]
    top_prob = probs[ind]
    top_prob = {i: top_prob[idx] for idx, i in enumerate(ind)}
    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}

    t = 0
    f = []
    pr = []
    for k, v in sorted_top_prob.items():
        t += v
        f.append(k)
        pr.append(v)
        if t >= p:
            break
    top_prob = np.array(pr) / np.sum(pr)
    token_id = np.random.choice(f, 1, p=top_prob)

    return int(token_id)


def generate(tokenizer, model, sentences, label, device):
    with torch.no_grad():
        for idx in range(sentences):
            finished = False
            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to(device)
            for i in range(100):
                outputs = model(cur_ids)
                logits = outputs.logits

                softmax_logits = torch.softmax(logits[0, -1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy())  # top-k-top-n sampling
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)

                if next_token_id in tokenizer.encode('  '):  # Ensure end of text token is handled
                    finished = True
                    break

            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)
            print(output_text)


def load_models(model_name, device):
    """
    Summary:
        Loading the trained model
    """
    print('Loading Trained GPT-2 Model')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.load_state_dict(torch.load(model_name, map_location=device))
    return tokenizer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inferencing Text Augmentation model')

    parser.add_argument('--model_name', default='mymodel.pt', type=str, action='store', help='Name of the model file')
    parser.add_argument('--sentences', type=int, default=5, action='store', help='Number of sentences in outputs')
    parser.add_argument('--label', type=str, action='store', help='Label for which to produce text', required=True)
    args = parser.parse_args()

    SENTENCES = args.sentences
    MODEL_NAME = args.model_name
    LABEL = args.label

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'

    TOKENIZER, MODEL = load_models(MODEL_NAME, DEVICE)

    MODEL.to(DEVICE)
    MODEL.eval()

    generate(TOKENIZER, MODEL, SENTENCES, LABEL, DEVICE)
