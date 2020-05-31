import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup, log_hyperparams
from transformers import T5Tokenizer, T5ForConditionalGeneration

try:
    from .utils import SummarizationDataset
except ImportError:
    from utils import SummarizationDataset

import files2rouge
from tqdm import tqdm 
import re
from pprint import pprint
import numpy as np 
import json
from os.path import join
import shutil
import time
import re
import logging
import numpy as np
import random
import string
import shutil
import files2rouge
import time

logger = logging.getLogger(__name__)


class SummarizationTrainer(BaseTransformer):

    mode = "language-modeling"

    def __init__(self, hparams):
        super().__init__(hparams, num_labels=None, mode=self.mode)
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
        )

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=80,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)

        return {"val_loss": loss, "preds": preds, "target": target}

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.save_pred_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.save_pred_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+") as p_writer, open(output_test_targets_file, "w+") as t_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
                t_writer.writelines(s + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        return self.test_end(outputs)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = SummarizationDataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        # Add BART specific options
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )
        parser.add_argument(
            "--logging_dir",
            default='tensorboard_logs',
            type=str,
            required=False,
            help="The directory for tensorboard_logs",
        )
        parser.add_argument(
            "--save_pred_dir",
            default='',
            type=str,
            required=False,
            help="Path to saved checkpoint"
        )
        parser.add_argument(
            "--visible_gpus",
            type=str
        )
        parser.add_argument(
            "--test_fname",
            default='test.hypo',
            type=str
        )
        parser.add_argument(
            "--ref_path",
            type=str
        )

        parser.add_argument(
            "--lenpen",
            type=float,
            default=1.0
        )
        parser.add_argument(
            "--test_epoch",
            type=int,
            default=-1
        )
        parser.add_argument(
            "--num_beams", '--beam',
            dest='num_beams',
            type=int,
            default=6
        )
        parser.add_argument(
            "--tune_decoder",
            action='store_true',
            default=False
        )
        parser.add_argument('--multitarget', action='store_true', default=False)
        parser.add_argument('--quick', action='store_true', default=False)
        parser.add_argument('--rouge_only', action='store_true', default=False, help='flag if you don\'t want to run predictions')
        parser.add_argument('--percentages', action='store_true', default=False, help='flag if you want to print as percentages')
        return parser

    def text_predictions(self, batch, device, args):
        dct = self.tokenizer.batch_encode_plus(batch, max_length=args.max_source_length, return_tensors="pt", pad_to_max_length=True)
        generated_ids = self.model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            max_length=30,
            repetition_penalty=2.5,
            length_penalty=args.lenpen,
            num_beams=args.num_beams,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.bos_token_id
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        return preds

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def predict(args, model, device, test_fname='test.hypo'):
    if not args.rouge_only:
        model.eval()
        model.freeze()
        input_path = os.path.join(args.data_dir, 'test.source')
        source_lns = [x.rstrip() for x in open(input_path).readlines()]
        example_batches = chunks(source_lns, args.eval_batch_size)
        outputs = []
        start = time.time()
        for example in tqdm(example_batches, total=int(len(source_lns)//args.eval_batch_size+1)):
            outputs += model.text_predictions(example, device, args)
        with open(os.path.join(args.save_pred_dir, test_fname), 'w') as fout:
            fout.write('\n'.join(outputs))
        end = time.time()
        print(f'Time to generate predictions: {end-start} sec\n\n')
    if args.ref_path:
        ref = args.ref_path
    elif args.multitarget:
        ref = os.path.join(args.data_dir, 'test-multitarget.jsonl')
    elif not quick:
        ref = os.path.join(args.data_dir, 'test.jsonl')
    else:
        ref = os.path.join(args.data_dir, 'test.target')
    
    cand = os.path.join(args.save_pred_dir, test_fname)
    # r = files2rouge.run(os.path.join(args.save_pred_dir, test_fname), ref, to_json=True)
    r = test_rouge(cand, ref, outpath=os.path.join(args.save_pred_dir, test_fname + '.rouge'), 
                    multitarget=args.multitarget, quick=args.quick)
    return r

def tune_decoder(args, model, device):
    lenpens = list(np.arange(0.2, 1.2, 0.2))
    lenpens = [round(l, 2) for l in lenpens]
    beams = list(range(2,7))

    best_r = None
    best_r1 = 0.
    best_lenpen = None
    best_beam = None

    os.makedirs(os.path.join(args.save_pred_dir, 'tuning'), exist_ok=True)

    print('tuning...')
    pbar = tqdm(total=len(lenpens)*len(beams))
    for l in lenpens:
        for b in beams:
            args.num_beams = b
            args.lenpen = l
            test_fname = f'tune-beam{b}-lenpen{l}.{args.test_fname}'
            r = predict(args, model, device, test_fname=test_fname)
            if r['rouge-1']['f'] > best_r1:
                best_r1 = r['rouge-1']['f']
                best_r = r
                best_lenpen = l
                best_beam = b
            pbar.update(1)
    print(f'Best lenpen: {best_lenpen} \t Best beam: {best_beam}')
    best_r['beam'] = best_beam
    best_r['lenpen'] = best_lenpen

    return best_r

def test_rouge(cand, ref, outpath=None, tmp_dir='/tmp/', multitarget=False, quick=False):
    print(cand)
    print(ref)
    print(multitarget, quick)

    def random_string(stringLength=8):
        """Generate a random string of fixed length """
        letters= string.ascii_lowercase
        return ''.join(random.sample(letters,stringLength))
    tmp_path = join(tmp_dir, 'tmp'+random_string())
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
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

def maybe_percentages(r, percentages):
    if percentages:
        for r_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            for m_type in ['f', 'p', 'r']:
                x = r[r_type][m_type]
                r[r_type][m_type] = x * 100
    return r

def main(args):

    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)


    model = SummarizationTrainer(args)
    model.tokenizer.add_special_tokens({"bos_token":"<s>"})
# print(SummarizationTrainer.tokenizer._extra_ids)
    trainer = generic_train(model, args)
    log_hyperparams(model)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.test_epoch != -1:
            checkpoints = [os.path.join(args.output_dir, f"checkpointepoch={args.test_epoch}.ckpt")]
        else:
            checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        print(f'loading checkpoint from {checkpoints[-1]}')
        # args.model_name_or_path = checkpoints[-1]
        # model = SummarizationTrainer(args).to(device)
        # import ipdb; ipdb.set_trace()
        # model = model.load_from_checkpoint(checkpoints[-1]) #.to(device)
        model.load_state_dict(torch.load(checkpoints[-1])['state_dict'])
        model = model.to(device)
        model.tokenizer.add_special_tokens({"bos_token":"<s>"})
        if args.tune_decoder:
            r = tune_decoder(args, model, device)
        else:
            r = predict(args, model, device, test_fname=args.test_fname)
            r['beam'] = args.num_beams
            r['lenpen'] = args.lenpen

        r = maybe_percentages(r, args.percentages)

        with open(os.path.join(args.save_pred_dir, f'{args.test_fname}.score'), 'w') as f:
            json.dump(r, f, indent=4)
    
        pprint(r)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SummarizationTrainer.add_model_specific_args(parser, os.getcwd())
    
    args = parser.parse_args()

    main(args)
