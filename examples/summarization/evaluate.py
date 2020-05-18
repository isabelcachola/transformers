import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from pprint import pprint
import os
from os.path import join
import shutil
import time
import re
import logging
import numpy as np
import json 
import random
import string
import shutil
import files2rouge
import time

def test_rouge(cand, ref, outpath=None, tmp_dir='/tmp/', multitarget=False, quick=False):
    print(cand)
    print(ref)
    def random_string(stringLength=8):
        """Generate a random string of fixed length """
        letters= string.ascii_lowercase
        return ''.join(random.sample(letters,stringLength))
    tmp_path = join(tmp_dir, 'tmp'+random_string())
    os.makedirs(tmp_path)
    # print(tmp_path)
    hyp_path = join(tmp_path, 'hyp.txt')
    ref_path = join(tmp_path, 'ref.txt')

    candidates = [line.strip().lower() for line in open(cand, encoding='utf-8')]
    if multitarget or not quick:
        references = [json.loads(line.strip())['target'] for line in open(ref, encoding='utf-8')]
    else:
        references = [line.lower().strip() for line in open(ref, encoding='utf-8')]
    assert len(candidates) == len(references), f'{tmp_dir}: len cand {len(candidates)} len ref {len(references)}'

    if quick and not multitarget:
        hyp = open(join(tmp_path, 'hyp.txt'), 'w')
        hyp.write('\n'.join([c.replace('\n', '') for c in candidates]))
        hyp.close()
        ref = open(join(tmp_path, 'ref.txt'), 'w')
        ref.write('\n'.join([r.lower().replace('\n', '') for r in references]))
        ref.close()
        _r = files2rouge.run(ref_path, hyp_path, to_json=True)
        return _r

    paper_ids = [json.loads(line.strip())['paper_id'] for line in open(ref, encoding='utf-8')]
    all_scores = []
    save_scores = []

    # For each prediction
    for cand_idx, cand in enumerate(candidates):
        curr_targets = references[cand_idx]
        curr_scores = []
        hyp = open(join(tmp_path, 'hyp.txt'), 'w')
        hyp.write(cand)
        hyp.close()
        # import ipdb; ipdb.set_trace()
        for tgt in curr_targets:
            tgt = tgt.lower().strip('\n')
            ref = open(join(tmp_path, 'ref.txt'), 'w')
            ref.write(tgt)
            ref.close()
            try:
                _r = files2rouge.run(ref_path, hyp_path, to_json=True)
            except Exception as e:
                print(e)
                exit(0)
            curr_scores.append(_r)
        # Take the max of curr scores
        r1 = [r['rouge-1']['f'] for r in curr_scores]
        max_idx = r1.index(max(r1))

        save_scores.append({
                        'paper_id': paper_ids[cand_idx],
                        'all_scores': curr_scores,
                        'max_idx': max_idx,
                        'prediction': cand,
                        'target': curr_targets
                            })
        all_scores.append(curr_scores[max_idx])
    # Average across all scores
    avg_scores = {"rouge-1": {
                    "f": [],
                    "p": [],
                    "r":[]
                    },
                "rouge-2": {
                    "f": [],
                    "p": [],
                    "r": []
                    },
                "rouge-l": {
                    "f": [],
                    "p": [],
                    "r": []
                    }
                }
    for score in all_scores:
        for r_type in score.keys():
            for m_type in score[r_type].keys():
                x = score[r_type][m_type]
                # print(x)
                avg_scores[r_type][m_type].append(x)
    #import ipdb; ipdb.set_trace()          
    for r_type in avg_scores.keys():
        for m_type in avg_scores[r_type].keys():
            x = avg_scores[r_type][m_type]
            avg_scores[r_type][m_type] = np.mean(x)

    if outpath:
        with open(outpath, 'w') as fout:
            for s in save_scores:
                fout.write(json.dumps(s) + '\n')

    shutil.rmtree(tmp_path)
    return avg_scores

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_summaries(lns, output_file_path, model_name_or_path, batch_size, device):
    output_file = Path(output_file_path).open("w")

    ckpt = torch.load(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained('t5-large')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)

    tokenizer = T5Tokenizer.from_pretrained('t5-large')

    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("summarization", {}))

    for batch in tqdm(list(chunks(lns, batch_size))):
        batch = [model.config.prefix + text for text in batch]

        dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="pt", pad_to_max_length=True)
        input_ids = dct["input_ids"].to(device)
        attention_mask = dct["attention_mask"].to(device)

        summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

        for hypothesis in dec:
            output_file.write(hypothesis + "\n")
            output_file.flush()

def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_path",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "data_dir", type=str, help="like cnn_dm/",
    )
    parser.add_argument(
        "--test_fname", type=str, default='test.hypo',
    )
    parser.add_argument(
        "--score_path", type=str, help="where to save the rouge score",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda", default=False, action='store_true', help="Whether to force the execution on CPU.",
    )
    parser.add_argument(
        "--quick", default=False, action='store_true'
    )
    parser.add_argument(
        "--multitarget", default=False, action='store_true'
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    args.output_path = join(args.data_dir, args.test_fname)
    if args.multitarget:
        args.reference_path = 'test-multitarget.jsonl'
    elif not args.quick:
        args.reference_path = 'test.jsonl'
    else:
        args.reference_path = 'test.target'
    args.reference_path = join(args.data_dir, args.reference_path)
    args.input_path = join(args.data_dir, 'test.source')

    source_lns = [x.rstrip() for x in open(args.input_path).readlines()]

    generate_summaries(source_lns, args.output_path, args.model_name_or_path, args.batch_size, args.device)

    # output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
    # reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

    # calculate_rouge(output_lns, reference_lns, args.score_path)

    test_rouge(args.output_path, args.reference_path, 
                multitarget=args.multitarget, 
                quick=args.quick)

if __name__ == "__main__":
    run_generate()
