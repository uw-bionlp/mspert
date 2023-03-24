import os
import json
import spacy

from tqdm import tqdm

from argparse import ArgumentParser

from spert import sampling
from spert.input_reader import JsonInputReader
from spert.evaluator import Evaluator
from spert import models
from spert.entities import Dataset
from spert.spert_trainer import SpERTTrainer
from transformers import BertTokenizer, BertConfig

import torch
from torch.utils.data import DataLoader


def main():
    args = parse_args()
    processor = None

    for doc_batch in batch(get_docs(args.dir), args.batch_size):
        del processor
        processor = DocumentProcessor(args)

        docs = [Doc(os.path.join(args.dir, d)) for d in doc_batch]
        predictions = processor.predict(docs, args.device)

        for id, preds in predictions.items():
            with open(os.path.join(args.dir, id + '.json'), 'w+', encoding='utf-8') as fout:
                fout.write(json.dumps(preds))


def get_docs(path):
    text_files = set([f for f in os.listdir(path) if f.endswith('.txt')])
    json_files = set([f.replace('.json', '.txt') for f in os.listdir(path) if f.endswith('.json')])
    return [t for t in text_files if t not in json_files]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dir', help='Absolute or relative path to the directory of .txt files to parse.')
    parser.add_argument('--batch_size', help='Number of files to process before reloading model.', default=20, type=int)
    parser.add_argument('--device', help='CUDA device to use. Defaults to -1 (CPU).', default=-1, type=int)
    parser.add_argument('--model_dir', help='Directory of model to use.', default=os.path.join('model','20230320'), required=False)

    return parser.parse_args()

class DocumentProcessor():

    def __init__(self, args):
        self.model_dir = args.model_dir
        self.spacy     = spacy.load('en_core_web_sm')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, do_lower_case=False, cache_dir=None)
        self.config    = BertConfig.from_pretrained(self.model_dir, cache_dir=None)

        self.args      = Args(os.path.join(self.model_dir, 'spert_args.json'))
        self.reader    = JsonInputReader(os.path.join(self.model_dir, 'spert_types.json'), self.tokenizer,
                                         max_span_size=self.args.max_span_size, logger=None)
        model_class    = models.get_model('spert')
        self.extractor = SpERTTrainer(self.args)
        self.model     = model_class.from_pretrained(self.model_dir,
                                            config = self.config,
                                            # SpERT model parameters
                                            cls_token = self.tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types = self.reader.relation_type_count - 1,
                                            entity_types = self.reader.entity_type_count,
                                            subtypes = self.reader.subtype_count,
                                            sent_label_types = self.reader.sent_type_count,
                                            max_pairs = self.args.max_pairs,
                                            prop_drop = self.args.prop_drop,
                                            size_embedding = self.args.size_embedding,
                                            freeze_transformer = self.args.freeze_transformer,
                                            subtype_classification = self.args.subtype_classification,
                                            concat_sent_pred = self.args.concat_sent_pred,
                                            include_adjacent = self.args.include_adjacent,
                                            include_word_piece_task = self.args.include_word_piece_task,
                                            concat_word_piece_logits = self.args.concat_word_piece_logits,
                                            cache_dir = self.args.cache_path)


    def predict(self, docs, device=-1):
        dataset = Dataset(label = 'eval', rel_types = self.reader._relation_types, entity_types = self.reader._entity_types,
                          subtypes = self.reader._subtypes, sent_types = self.reader._sent_types, neg_entity_count = self.reader._neg_entity_count,
                          neg_rel_count = self.reader._neg_rel_count, max_span_size = self.reader._max_span_size)
        for doc in docs:
            self.parse_document(dataset, self.tokenizer, doc)

        evaluator = Evaluator(dataset, self.reader, self.tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, predictions_path=None,
                              examples_path=None, example_count=0)
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                 num_workers=1, collate_fn=sampling.collate_fn_padding)
        
        with torch.no_grad():
            self.model.eval()
            if device != -1:
                self.model.to(device)

            # iterate batches
            for batch in tqdm(data_loader, total=dataset.document_count):

                # move batch to selected device
                if device != -1:
                    batch = self.to_device(batch, device)

                # run model (forward pass)
                result = self.model( \
                                encodings = batch['encodings'],
                                context_masks = batch['context_masks'],
                                entity_masks = batch['entity_masks'],
                                entity_masks_adj = batch['entity_masks_adj'],
                                entity_sizes = batch['entity_sizes'],
                                entity_spans = batch['entity_spans'],
                                entity_sample_masks = batch['entity_sample_masks'],
                                inference = True)
                entity_clf, subtype_clf, rel_clf, rels, sent_clf = result
    
                # evaluate batch
                evaluator.eval_batch( \
                                    batch_entity_clf = entity_clf,
                                    batch_subtype_clf = subtype_clf,
                                    batch_rel_clf = rel_clf,
                                    batch_rels = rels,
                                    batch_sent_clf = sent_clf,
                                    batch = batch)

        predictions = evaluator.store_predictions()
        predictions_dict = {}
        for prediction in predictions:
            doc_id = prediction['doc_id'].split('_')[0]
            prediction['sentence_idx'] = int(prediction['doc_id'].split('_')[1])
            if doc_id in predictions_dict:
                predictions_dict[doc_id]['sentences'].append(prediction)
            else:
                predictions_dict[doc_id] = { 'id': doc_id, 'sentences': [ prediction ] }
        
        return predictions_dict


    def parse_document(self, dataset, tokenizer, doc):
        tokenized = self.spacy(doc.text)

        for sent_idx, sentence in enumerate(tokenized.sents):
            doc_tokens = []
            doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

            for i, spacy_token in enumerate(sentence):
                token_phrase = spacy_token.text
                token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)
                if not token_encoding:
                    token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
                span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
                token = dataset.create_token(i, span_start, span_end, token_phrase)
                doc_tokens.append(token)
                doc_encoding += token_encoding
            doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]

            dataset.create_document(doc_id=doc.name + '_' + str(sent_idx), tokens=doc_tokens, entity_mentions=[], subtype_mentions=[],
                                    relations=[], sent_labels=[0], word_piece_labels=[('None', 0)]*len(doc_encoding),
                                    doc_encoding = doc_encoding)
        
        return dataset


    def to_device(self, batch, device):
        converted_batch = dict()
        for key in batch.keys():
            converted_batch[key] = batch[key].to(device)

        return converted_batch


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class Doc:
    def __init__(self, path):
        self.name = path.split(os.path.sep)[-1].split('.')[0]
        self.path = path

        with open(path, 'r', encoding='utf-8') as fin:
            self.text = fin.read()
        

class Args:
    def __init__(self, path):
        with open(path, encoding='utf-8') as fin:
            d = json.loads(fin.read())
        for key, value in d.items():
            setattr(self, key, value)


main()