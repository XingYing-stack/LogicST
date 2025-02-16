import copy

import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss, NCRLoss
import numpy as np
import dill as pickle
from scipy.stats import norm
import random
from collections import  defaultdict, deque
import json
from mine_rule import transform_negations
from tqdm import tqdm



class Logical_Pseudo_Labeling_Module(nn.Module):
    def __init__(self, config, rule_path = None, minC=0.9, rel2id=None, rel_info=None, indicator='pos', top_K=25, device="cuda:0", lamb=0.995, eta=0.1, gamma=10, dataset='docred'):
        super().__init__()
        self.config = config
        self.rel2id = self.add_inverse(rel2id)

        if dataset == 'docred':
            rel_info.update({'anti_' + k : 'anti_' + v for k, v in rel_info.items()})
            self.rel_info = {v: k for k,v in rel_info.items()}
        else:
            self.rel_info = {name:name for name in self.rel2id.keys()}

        if rule_path:
            f = open(rule_path, 'rb')
            self.rules = pickle.load(f)
            self.minC = minC
            self.filter_rules()


        self.diagnose_mode = config.diagnose_mode # vanilla / rule
        self.sampling_mode = config.sampling_mode  # uniform / likelihood / best / worst
        self.ratio_pos, self.ratio_neg = config.ratio_pos, config.ratio_neg
        self.relid2rel = {v:k for k,v in self.rules.rel2id.items()}
        self.relation_num = len(self.relid2rel) // 2
        self.indicator = indicator
        self.top_K = top_K

        # =======GAPs BANK========
        self.GAPS_BANK = torch.zeros(config.num_labels - 1, ).to(device)
        self.lamb = lamb
        self.eta = eta
        self.gamma = gamma
        print('gamma is:', gamma)

        self.size_para = 5


    def add_inverse(self, rel2id):
        self.relation_num = len(rel2id) - 1
        anti_dict = {}
        length = len(rel2id)
        for rel, id in rel2id.items():
            if rel.lower() != 'na':
                anti_dict['anti_' + rel] = id + length - 1
        rel2id.update(anti_dict)
        return rel2id

    def update_bank(self, logits):
        gaps = logits[:, 1:] - logits[:, 0].unsqueeze(-1) # bigger is better
        if torch.sum(self.GAPS_BANK) == 0:
            self.GAPS_BANK = gaps.mean(0)
        else:
            self.GAPS_BANK = self.lamb * self.GAPS_BANK + (1-self.lamb) * gaps.mean(0)

    def print_bank(self):
        print('Bank:', self.GAPS_BANK.tolist())

    def filter_rules(self):
        print('=' * 50)
        print('RULEs:')
        self.rule_body, self.rule_head = [], []
        for old_rule in self.rules.rules:
            if 'anti' in old_rule.relation_name:
                continue
            for conf, body, body_name, hc in zip(old_rule.confidence_lst, old_rule.body_lst, old_rule.body_lst_NAMES, old_rule.hc_lst):
                if conf >= self.minC:
                    self.rule_body.append([self.rel2id.get(self.rel_info.get(name, None), None) for name in body_name] )
                    self.rule_head.append( self.rel2id.get(self.rel_info.get(old_rule.relation_name, None), None))
                    print(body_name, '->', old_rule.relation_name, ':', conf)
                    assert None not in self.rule_body[-1]
                    assert None not in self.rule_head
        if type(self.rules.negations) == list:
            self.negations = {self.rel2id[negation.relation_name]  : negation for negation in transform_negations(self.rules.negations, new_rel2id=self.rel2id)}
        elif type(self.rules.negations) == dict:
            self.negations = self.rules.negations
        print('=' * 50)

    # Determine whether there is a path consisting of body, and return a list consisting of all paths that satisfy body_list
    def path_find(self, body_lst, adjcent_matrix, entity_num_list):
        result = []
        # BFS Search
        for start in entity_num_list:
            queue = deque([ [start, [start]] ] )
            for i in range(len(body_lst)):
                body = body_lst[i]
                for _ in range(len(queue)):
                    node, path = queue.popleft()
                    for next_node in adjcent_matrix[node][body]:
                        queue.append([next_node, path + [next_node]])
            while len(queue) > 0:
                result.append(queue.popleft()[-1])
        return result



    def add_composition_candidate(self, body_lst, target, adjcent_matrix, entity_num_list, candidates_lst,
                                  gaps, pseudo_label, hts_to_index, epoch=-1):
        pathes = self.path_find(body_lst, adjcent_matrix, entity_num_list)

        for path in pathes:
            temp_candidates_lst = list()

            start, end = path[0], path[-1]
            if end in adjcent_matrix[start][target]:
                continue
            # points are flipped to resolve the conflict



            points = {(path[0], target, path[-1])}
            for i in range(len(body_lst)):
                if body_lst[i] <= self.relation_num: # 非anti
                    point = (path[i], body_lst[i], path[i+1])
                else:# anti
                    point = (path[i+1], body_lst[i] - self.relation_num, path[i])
                points.add(point)
            for candidates in candidates_lst:
                if len(candidates & points) > 0:
                    temp_candidates_lst.append(candidates)
                else:
                    for point in points:
                        temp_candidates_lst.append(candidates | {point})
            # Remove Super-sets
            result = []

            for s in temp_candidates_lst:
                is_superset = False
                for other_set in temp_candidates_lst:
                    if s != other_set and s.issuperset(other_set):
                        is_superset = True
                        break
                if not is_superset:
                    result.append(s)

            candidates_lst = self.filter_and_sort_candidates(result, gaps, pseudo_label, hts_to_index, epoch=epoch)
        return candidates_lst

    def add_negation_candidate(self, negation, candidates_lst, relation_to_entity, adjcent_matrix
                               , gaps, pseudo_label, hts_to_index, epoch):
        rel_id = self.rel2id[negation.relation_name]
        points_list = []
        for head, tail in zip(relation_to_entity[rel_id]['head'], relation_to_entity[rel_id]['tail']):
            for neg_rel_id in negation.NEG_head_entity_AS_HEAD:
                for next_tail in adjcent_matrix[head][neg_rel_id]:
                    points_list.append({(head, rel_id, tail), (head, neg_rel_id, next_tail) } )
            for neg_rel_id in negation.NEG_head_entity_AS_TAIL:
                for next_head in adjcent_matrix[head][neg_rel_id + self.relation_num]:
                    points_list.append({(head, rel_id, tail), (next_head, neg_rel_id, head ) })

            for neg_rel_id in negation.NEG_tail_entity_AS_HEAD:
                for next_tail in adjcent_matrix[tail][neg_rel_id]:
                    points_list.append({(head, rel_id, tail), (tail, neg_rel_id, next_tail)})

            for neg_rel_id in negation.NEG_tail_entity_AS_TAIL:
                for next_head in adjcent_matrix[tail][neg_rel_id + self.relation_num]:
                    points_list.append({(head, rel_id, tail), (next_head, neg_rel_id, tail)})

        for points in points_list:
            temp_candidates_lst = []
            for candidates in candidates_lst:
                if len(candidates & points) > 0:
                    temp_candidates_lst.append(candidates)
                else:
                    for point in points:
                        temp_candidates_lst.append(candidates | {point})

            result = []
            for s in temp_candidates_lst:
                is_superset = False
                for other_set in temp_candidates_lst:
                    if s != other_set and s.issuperset(other_set):
                        is_superset = True
                        break
                if not is_superset:
                    result.append(s)

            candidates_lst = self.filter_and_sort_candidates(result, gaps, pseudo_label, hts_to_index, epoch)
        return candidates_lst

    def estimate_candidate(self, candidate, gaps, original_pseudo_labels, hts_to_index, epoch):
        flip_coordinates = torch.LongTensor([[hts_to_index[point[0], point[-1]], point[1]] for point in candidate if point[0] != point[-1]]).to(original_pseudo_labels.device)
        if flip_coordinates.shape[0] == 0:
            return float('-inf')
        flip_coordinates_pseudo_label = original_pseudo_labels[flip_coordinates[:, 0], flip_coordinates[:, 1]]

        if self.indicator == 'pos':
            return (flip_coordinates_pseudo_label == 0).sum().item()
        elif self.indicator == 'min':
            return -flip_coordinates_pseudo_label.shape[0]
        elif self.indicator == 'prob':
            flip_coordinates_gaps = gaps[flip_coordinates[:, 0], flip_coordinates[:, 1] - 1]

            flip_coordinates_being_true = torch.log(torch.sigmoid(flip_coordinates_gaps / self.size_para))
            flip_coordinates_being_false = torch.log(1 - torch.sigmoid(flip_coordinates_gaps / self.size_para))
            flip_coordinates_probability = flip_coordinates_pseudo_label * flip_coordinates_being_false + (1 - flip_coordinates_pseudo_label) * flip_coordinates_being_true
            return torch.sum(flip_coordinates_probability).item() + (flip_coordinates_pseudo_label == 0).sum().item() / (1 + epoch) * np.log(self.gamma)
        else:
            raise Exception('NOT IMPLEMENTED.')


    def filter_and_sort_candidates(self, candidates, gaps, pseudo_label, hts_to_index, epoch):
        if len(candidates) > 1:
            measured_indicators = [self.estimate_candidate(candidate, gaps, pseudo_label, hts_to_index, epoch) for candidate in candidates]
            sorted_candidates = sorted(zip(measured_indicators, candidates), reverse=True)

            return [element[1] for element in sorted_candidates][:self.top_K]
        else:
            return candidates



    def diagnose(self, pseudo_labels, logits,  hts_list, labels, epoch=0):
        gaps = logits[:, 1:] - logits[:, 0].unsqueeze(-1)
        if ((gaps < 0) & (gaps != -100)).sum() > 0:
            NEG_TH = np.quantile(gaps[(gaps < 0) & (gaps != -100)].cpu().numpy(), self.ratio_neg)
        else:
            NEG_TH = -100

        if ((gaps > 0) & (gaps != 100)).sum() > 0:
            POS_TH = np.quantile(gaps[(gaps > 0) & (gaps != 100)].cpu().numpy(), 1 - self.ratio_pos)
        else:
            POS_TH = 100
        # NOTE: to remind that instance_mask having different id!!! i.e., id -1
        pseudo_instance_masks = (gaps <= NEG_TH) | (gaps >= POS_TH)

        # -----------------------------
        gaps = gaps - self.eta * self.GAPS_BANK
        # -----------------------------

        if self.diagnose_mode == 'vanilla': # do nothing
            return [[pseudo_labels, pseudo_instance_masks] ]
        else:
            pseudo_labels_list = torch.split(pseudo_labels, [len(_) for _ in hts_list], dim=0)
            pseudo_instance_masks_list = torch.split(pseudo_instance_masks, [len(_) for _ in hts_list], dim=0)

            diagnosed_labels_list, diagnoses_instance_masks_list = [], []
            for i in range(len(hts_list)): # i-th document
                pseudo_label, hts = pseudo_labels_list[i], hts_list[i]
                pseudo_instance_mask = pseudo_instance_masks_list[i] # ready to flip only set to True!

                diagnosed_labels, diagnoses_instance_masks = [], []
                hts_to_index = {tuple(ht):index for index, ht in enumerate(hts)}
                ht_and_relation = torch.nonzero(pseudo_label, as_tuple=False)
                ht_and_relation = ht_and_relation[ht_and_relation[:, 1] != 0].tolist()
                pseudo_triplet = defaultdict(lambda: defaultdict(list))
                pseudo_relation_to_entity = defaultdict(lambda: defaultdict(list))

                max_entity_num = max([_[0] for _ in hts])
                for ht, relation in ht_and_relation:
                    h, t = hts[ht]
                    pseudo_triplet[h][relation].append(t)
                    pseudo_triplet[t][relation + self.relation_num].append(h)

                    pseudo_relation_to_entity[relation]['head'].append(h)
                    pseudo_relation_to_entity[relation]['tail'].append(t)

                    pseudo_relation_to_entity[relation + self.relation_num]['head'].append(t)
                    pseudo_relation_to_entity[relation + self.relation_num]['tail'].append(h)

                candidates = [set()]

                for body_lst, head in zip(self.rule_body, self.rule_head):
                    flag = True
                    for body in body_lst:
                        if body not in pseudo_relation_to_entity.keys():
                            flag = False
                            break
                    if flag == False:
                        continue
                    candidates = self.add_composition_candidate(body_lst, head, pseudo_triplet, range(0, max_entity_num +1 ), candidates
                                                                ,gaps ,pseudo_label ,hts_to_index, epoch=epoch)


                # negation rules
                for relation_id in pseudo_relation_to_entity.keys():
                    if relation_id <= self.relation_num:
                        negation = self.negations[relation_id]
                        candidates = self.add_negation_candidate(negation, candidates, pseudo_relation_to_entity, pseudo_triplet
                                                                 ,gaps ,pseudo_label, hts_to_index, epoch=epoch)


                candidates = self.filter_and_sort_candidates(candidates, gaps, pseudo_label, hts_to_index, epoch=epoch)
                # build labels
                for candidate in candidates:
                    # build coordinates to flip
                    # bug here! not to flip same h & t
                    flip_coordinates = torch.LongTensor([[hts_to_index[point[0], point[-1]], point[1]] for point in candidate if point[0] != point[-1]]).to(pseudo_label.device)
                    # Create labels to flip
                    diagnose_label, diagnose_instance_mask = pseudo_label.clone(), pseudo_instance_mask.clone()
                    # flip
                    if flip_coordinates.shape[0] > 0:
                        diagnose_label[flip_coordinates[:, 0], flip_coordinates[:, 1]] = 1 - diagnose_label[flip_coordinates[:, 0], flip_coordinates[:, 1]]
                        diagnose_label[:, 0]  = (diagnose_label[:, 1:].sum(1) == 0.).to(logits)
                        # NOTE: -1 !!!
                        diagnose_instance_mask[flip_coordinates[:, 0], flip_coordinates[:, 1] - 1] = True
                    # append
                    diagnosed_labels.append(diagnose_label)
                    diagnoses_instance_masks.append(diagnose_instance_mask)
                diagnosed_labels_list.append(diagnosed_labels)
                diagnoses_instance_masks_list.append(diagnoses_instance_masks)

            return diagnosed_labels_list, diagnoses_instance_masks_list


    def sample_pseudo_label(self, NEW_pseudo_labels_with_instance_MASK, logits, hts):
        # Note that the args format of 'vanilla' is different from 'logic'
        if self.diagnose_mode == 'vanilla':
            return random.sample(NEW_pseudo_labels_with_instance_MASK, 1)[0]
        else:
            diagnose_pseudo_labels_lst, diagnose_instance_MASK_lst = NEW_pseudo_labels_with_instance_MASK
            if self.sampling_mode == 'uniform':
                specific_pseudo_label = torch.cat([random.sample(diagnose_pseudo_labels, 1)[0]  for diagnose_pseudo_labels in diagnose_pseudo_labels_lst], dim=0)
                specific_instance_MASK = torch.cat([random.sample(diagnose_instance_MASK, 1)[0]  for diagnose_instance_MASK in diagnose_instance_MASK_lst], dim=0)

                return specific_pseudo_label, specific_instance_MASK
            elif self.sampling_mode == 'best':
                specific_pseudo_label = torch.cat([diagnose_pseudo_labels[0]  for diagnose_pseudo_labels in diagnose_pseudo_labels_lst], dim=0)
                specific_instance_MASK = torch.cat([diagnose_instance_MASK[0]  for diagnose_instance_MASK in diagnose_instance_MASK_lst], dim=0)

                return specific_pseudo_label, specific_instance_MASK

            elif self.sampling_mode == 'worst':
                specific_pseudo_label = torch.cat([diagnose_pseudo_labels[-1]  for diagnose_pseudo_labels in diagnose_pseudo_labels_lst], dim=0)
                specific_instance_MASK = torch.cat([diagnose_instance_MASK[-1]  for diagnose_instance_MASK in diagnose_instance_MASK_lst], dim=0)

                return specific_pseudo_label, specific_instance_MASK

            else:
                raise Exception('not implemented!')

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1, args=None):
        super().__init__()
        self.config = config
        # BERT model
        self.model = model
        self.hidden_size = config.hidden_size
        self.args = args

        if self.args.loss_fn == "ATL":
            self.loss_fnt = ATLoss()
        elif self.args.loss_fn == "NCRL":
            self.loss_fnt = NCRLoss(isReg=bool(args.NCRL_REG))

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        # ------------
        self.negative_sampling_ratio = self.config.negative_sampling_ratio

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    # sequence_output : tensor((batch_size, sequence_length, hidden_size))  attention : tensor( (batch_size, num_heads, sequence_length, sequence_length) )
    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size() # (batch_size, num_heads, sequence_length, sequence_length)
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def pseudo_label(self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            entity_pos=None,
            hts=None,
            instance_mask=None,
            discover=False,
            epoch=-1
            ):
        # sequence_output : tensor((batch_size, sequence_length, hidden_size))
        # attention : tensor( (batch_size, num_heads, sequence_length, sequence_length) )
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        if isinstance(labels, list):
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)

        logits_old = torch.clone(logits)
        MASK = labels[:, 0] == 0
        logits[MASK, 1:] = logits[MASK, 1:] - logits[MASK, 0].unsqueeze(-1)
        logits[MASK, 0] = 0.0

        logits[MASK, 1:] = torch.where(labels[MASK, 1:] == 1, torch.tensor(100.0).to(logits), logits[MASK, 1:])

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)  # confidence thresholding based pseudo label

        NEW_pseudo_labels_with_instance_MASK = self.logical_pseudo_labeling_module.diagnose(output[0], logits, hts, labels, epoch=epoch)
        specific_pseudo_label, instance_mask = self.logical_pseudo_labeling_module.sample_pseudo_label(NEW_pseudo_labels_with_instance_MASK, logits, hts)

        return specific_pseudo_label, instance_mask, logits_old


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                discover = False,
                epoch=-1  # 当前训练的epoch
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        if discover:
            return logits.detach().cpu().numpy()
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),) # confidence thresholding based pseudo label
        if labels is not None:
            if isinstance(labels, list):
                labels = [torch.tensor(label) for label in labels]
                labels = torch.cat(labels, dim=0).to(logits)
            if instance_mask is None:
                instance_mask = torch.ones_like(labels).to(labels)[:, 1:] # default selecting all instances

            MASK = labels[:, 0] == 0
            labels_ann, logits_ann = labels[MASK], logits[MASK]
            loss_ann = self.loss_fnt(logits_ann.float(), labels_ann.float(), instance_mask[MASK])
            ann_number = instance_mask[MASK].sum().item() # 可能差一个常数

            # negative sampling
            if self.negative_sampling_ratio > 0:
                MASK = (labels[:, 0] == 1) & (torch.rand(labels.size(0), ).to(logits) < self.negative_sampling_ratio)
                ns_number = instance_mask[MASK].sum().item()
                ns_loss = self.loss_fnt(logits[MASK].float(), labels[MASK].float(), instance_mask[MASK])

                if self.config.norm_by_sample:
                    loss = (ann_number / (ann_number + ns_number + 1e-5)) * loss_ann  + (ns_number / (ann_number + ns_number + 1e-5)) * ns_loss
                else:
                    loss = loss_ann + ns_loss
            else:
                loss = loss_ann
            # ======================
            output = (loss.to(sequence_output),) + output + (logits,)
        return output