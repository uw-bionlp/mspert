import random
from collections import OrderedDict
import torch

from spert import util


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int, subtype_types: dict):
    encodings = doc.encoding
    token_count = len(doc.tokens)

    # BERT sequence length in word_pieces
    context_size = len(encodings)

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    pos_entity_masks_adj = []
    for e in doc.entities:

        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))


        pos_entity_masks_adj.append(create_adjacent_entity_mask(e, doc.tokens, context_size))

        pos_entity_sizes.append(len(e.tokens))

    pos_subtypes = OrderedDict([(k, []) for k in subtype_types.keys()])
    for subtypes in doc.subtypes:
        # pos_subtypes.append(s.entity_type.index)
        for layer_name, s in subtypes.items():
            pos_subtypes[layer_name].append(s.entity_type.index)


    # positive relations

    # collect relations between entity pairs
    entity_pair_relations = dict()
    for rel in doc.relations:
        pair = (rel.head_entity, rel.tail_entity)
        if pair not in entity_pair_relations:
            entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel)

    # build positive relation samples
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair
        s1, s2 = head_entity.span, tail_entity.span
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        pos_rel_spans.append((s1, s2))

        pair_rel_types = [r.relation_type.index for r in rels]
        pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]
        pos_rel_types.append(pair_rel_types)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))

    # negative entities
    neg_entity_spans, neg_entity_sizes = [], []
    neg_entity_tokens = []
    neg_entity_spans_adj = []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span


            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

                previous_span = doc.tokens[i-1].span if i > 0 else None
                next_span = doc.tokens[i+size].span if i+size < token_count else None
                neg_entity_spans_adj.append((previous_span, next_span))

    # sample negative entities
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes, neg_entity_spans_adj)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes, neg_entity_spans_adj = zip(*neg_entity_samples) if neg_entity_samples else ([], [], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_masks_adj = [create_adjacent_entity_mask_from_spans(s, e, context_size) for s, e in neg_entity_spans_adj]
    assert len(neg_entity_masks) == len(neg_entity_masks_adj)




    neg_entity_types = [0] * len(neg_entity_spans)

    # neg_subtypes = [0] * len(neg_entity_spans)
    neg_subtypes = OrderedDict([(k, [0] * len(neg_entity_spans)) \
                                                for k in subtype_types.keys()])





    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [(0,) * (rel_type_count-1)] * len(neg_rel_spans)

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_masks_adj = pos_entity_masks_adj + neg_entity_masks_adj
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    # subtypes = pos_subtypes + neg_subtypes
    subtypes = OrderedDict([(k, []) for k in subtype_types.keys()])
    for k in subtypes.keys():
        subtypes[k] = pos_subtypes[k] + neg_subtypes[k]

    rels = pos_rels + neg_rels
    rel_types = pos_rel_types + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    for k, v in subtypes.items():
        assert len(entity_types) == len(v)
    assert len(entity_masks) == len(entity_masks_adj)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_masks_adj = torch.stack(entity_masks_adj)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)

        # subtypes = torch.tensor(subtypes, dtype=torch.long)
        for k, v in subtypes.items():
            subtypes[k] = torch.tensor(v, dtype=torch.long)
        subtypes = torch.stack(list(subtypes.values()) ,dim=1)

    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_masks_adj = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

        # subtypes = torch.zeros([1], dtype=torch.long)
        for k, v in subtypes.items():
            subtypes[k] = torch.zeros([1], dtype=torch.long)
        subtypes = torch.stack(list(subtypes.values()) ,dim=1)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.float32)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count-1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)



    sent_labels = torch.tensor(doc.sent_labels, dtype=torch.float32)
    word_piece_labels = torch.tensor(doc.word_piece_label_idxs, dtype=torch.long)


    return dict(encodings=encodings, context_masks=context_masks,
                entity_masks=entity_masks, entity_masks_adj=entity_masks_adj,
                entity_sizes=entity_sizes, entity_types=entity_types,
                subtypes=subtypes,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks,
                sent_labels=sent_labels, word_piece_labels=word_piece_labels)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []
    entity_masks_adj = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

            previous_span = doc.tokens[i-1].span if i > 0 else None
            next_span = doc.tokens[i+size].span if i+size < token_count else None
            entity_masks_adj.append(create_adjacent_entity_mask_from_spans(previous_span, next_span, context_size))

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_masks_adj = torch.stack(entity_masks_adj)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_masks_adj = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks,
                entity_masks=entity_masks, entity_masks_adj=entity_masks_adj,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks)


def create_entity_mask(start, end, context_size):

    # (context_size)
    mask = torch.zeros(context_size, dtype=torch.bool)

    # (context_size)
    mask[start:end] = 1

    return mask


def create_adjacent_entity_mask(entity, tokens, context_size):


    previous_token_idx = entity._tokens[0]._index - 1
    next_token_idx =     entity._tokens[-1]._index + 1

    if previous_token_idx < 0:
        previous_mask = torch.zeros(context_size, dtype=torch.bool)
    else:
        span =  tokens[previous_token_idx].span
        previous_mask = create_entity_mask(*span, context_size)

    if next_token_idx >= len(tokens):
        next_mask = torch.zeros(context_size, dtype=torch.bool)
    else:
        span =  tokens[next_token_idx].span
        next_mask =  create_entity_mask(*span, context_size)

    mask = torch.logical_or(previous_mask, next_mask)

    return mask


def create_adjacent_entity_mask_from_spans(previous_span, next_span, context_size):

    if previous_span is None:
        previous_mask = torch.zeros(context_size, dtype=torch.bool)
    else:
        previous_mask = create_entity_mask(*previous_span, context_size)

    if next_span is None:
        next_mask = torch.zeros(context_size, dtype=torch.bool)
    else:
        next_mask =  create_entity_mask(*next_span, context_size)

    mask = torch.logical_or(previous_mask, next_mask)

    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)

    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
