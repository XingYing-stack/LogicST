from tqdm import tqdm
import ujson as json
import torch
import numpy as np
import copy
import dill as pickle
import argparse
from collections import defaultdict

docred_rel_info = json.load(open('./dataset/docred/rel_info.json', 'r'))


class Rule:
    def __init__(self, target: int, rel_name: str):
        self.body_lst = []
        self.body_lst_NAMES = []
        self.confidence_lst = []  # closed-world-confidence
        self.hc_lst = []  # head-coverage
        self.target = target
        self.relation_name = rel_name

    def append(self, new_body, new_cofidence, new_hc):
        self.body_lst.append(new_body)
        self.confidence_lst.append(new_cofidence)
        self.hc_lst.append(new_hc)

class Negation:
    def __init__(self, rel_name : str):
        self.relation_name = rel_name
        self.head_entity_possible_types = set()
        self.tail_entity_possible_types = set()
        self.NEG_head_entity_AS_HEAD = set()
        self.NEG_head_entity_AS_TAIL = set()
        self.NEG_tail_entity_AS_HEAD = set()
        self.NEG_tail_entity_AS_TAIL = set()

    def build_entity_types(self, relation_to_types):
        self.head_entity_possible_types = relation_to_types['HEAD'][self.relation_name]
        self.tail_entity_possible_types = relation_to_types['TAIL'][self.relation_name]

    def build_negations(self, types_to_relations, rel2id):
        # ----------------
        self.NEG_head_entity_AS_HEAD = {rel for rel, id in rel2id.items() if 'anti' not in rel.lower() and 'na' not in rel.lower()}
        for type in self.head_entity_possible_types:
            self.NEG_head_entity_AS_HEAD -= types_to_relations['HEAD'][type]
        # ----------------
        self.NEG_head_entity_AS_TAIL = {rel for rel, id in rel2id.items() if 'anti' not in rel.lower() and 'na' not in rel.lower()}
        for type in self.head_entity_possible_types:
            self.NEG_head_entity_AS_TAIL -= types_to_relations['TAIL'][type]
        # ----------------
        self.NEG_tail_entity_AS_HEAD = {rel for rel, id in rel2id.items() if 'anti' not in rel.lower() and 'na' not in rel.lower()}
        for type in self.tail_entity_possible_types:
            self.NEG_tail_entity_AS_HEAD -= types_to_relations['HEAD'][type]
        # ----------------
        self.NEG_tail_entity_AS_TAIL = {rel for rel, id in rel2id.items() if 'anti' not in rel.lower() and 'na' not in rel.lower()}
        for type in self.tail_entity_possible_types:
            self.NEG_tail_entity_AS_TAIL -= types_to_relations['TAIL'][type]
        # ----------------


class RuleMiner:
    def __init__(self, file_in="./dwie/train_annotated.json", max_rule_length=2, minHC=0.008,
                 minC=0.1, minBodyInstance=3, rel2id=None, device='cpu'):  # todo:test hyper parameter
        self.max_rule_length = max_rule_length
        self.minHC = minHC
        self.minC = minC
        self.minBodyInstance = minBodyInstance
        with open(file_in, "r") as fh:
            data = json.load(fh)
        self.data = data

        rel2id = sorted(rel2id.items(), key=lambda kv: (kv[1], kv[0]))
        rel2id = {kv[0]: kv[1] for kv in rel2id}
        self.rel2id = self.add_inverse(rel2id)
        self.device = device
        if 'docred' in file_in:
            self.pid2name = lambda x: docred_rel_info[x] if 'anti' not in x else ('anti_' + docred_rel_info[x.split(
                '_')[
                -1]])
        elif 'dwie' in file_in or 'chemdisgene' in file_in:
            self.pid2name = lambda x: x
        self.rules = [Rule(id, self.pid2name(rel)) for rel, id in self.rel2id.items() if rel.lower() != 'na']

        id2entity, entity2id, facts = self.transform()
        facts = torch.tensor(facts).to(self.device).int()
        facts = facts[facts[:, 0] != facts[:, -1], :]
        self.id2entity = id2entity
        self.entity2id = entity2id
        self.facts = facts
        self.facts_target = [self.facts[self.facts[:, 1] == target, :] for rel, target in self.rel2id.items() if
                             rel.lower() != 'na']
        self.facts_target_NameAndID = [(rel, id) for rel, id in self.rel2id.items() if rel.lower() != 'na']
        # list(int)
        self.facts_target_size = [fact.size(0) for fact in self.facts_target]

    def add_inverse(self, rel2id):
        anti_dict = {}
        k = 0
        length = len(rel2id)
        for rel, id in rel2id.items():
            if rel.lower() != 'na':
                anti_dict['anti_' + rel] = k + length
                k += 1
        rel2id.update(anti_dict)
        return rel2id

    def transform(self):
        len_relation = len(self.rel2id) // 2
        id2entity = {}
        entity2id = {}  

        facts = set()  
        k = 0
        for doc_id, sample in tqdm(enumerate(self.data), desc="Example"):
            if 'vertexSet' in sample:
                entities = sample['vertexSet']
                for entityid, entity in enumerate(entities):
                    id2entity[k] = [str(doc_id) + '_' + str(entityid),
                                    set([entity[i]['name'] for i in range(len(entity))])]
                    entity2id[str(doc_id) + '_' + str(entityid)] = k
                    k += 1
            elif 'entity' in sample:
                entities = sample['entity']
                for entity in entities:
                    entityID_lst = entity['id'].split('|')
                    for entityID in entityID_lst:
                        id2entity[k] = str(doc_id) + '_' + str(entityID)
                        entity2id[str(doc_id) + '_' + str(entityID)] = k
                        k += 1
            else:
                raise Exception

            if 'vertexSet' in sample:
                labels = sample['labels']
                for label in labels:
                    h = entity2id[str(doc_id) + '_' + str(label['h'])]
                    t = entity2id[str(doc_id) + '_' + str(label['t'])]
                    r_real = self.rel2id[label['r']]
                    r_anti = r_real + len_relation
                    facts.add((h, r_real, t))
                    facts.add((t, r_anti, h))
            elif 'entity' in sample:
                labels = sample['relation']
                for label in labels:
                    try:
                        h = entity2id[str(doc_id) + '_' + str(label['subj'])]
                        t = entity2id[str(doc_id) + '_' + str(label['obj'])]
                        r_real = self.rel2id[label['type']]
                        r_anti = r_real + len_relation
                        facts.add((h, r_real, t))
                        facts.add((t, r_anti, h))
                    except:
                        print('Entity not exist.')
                        pass
            else:
                raise Exception

        return id2entity, entity2id, sorted(list(facts))

    def calculate(self, facts_1, facts_2):
        if facts_2.size(0) == 0 or facts_2.size(0) == 0:
            return (-1, -1)

        correct = 0
        for i in range(facts_1.size(0)):
            h, *_, t = facts_1[i]
            is_correct = torch.sum((facts_2[:, 0] == h) & (facts_2[:, -1] == t)).item()
            correct += is_correct
            if is_correct > 1:
                raise Exception('Duplicate!')
        return (correct / facts_1.size(0), correct / facts_2.size(0))

    def calculate_tensor(self, facts_1, facts_2):
        if facts_2.size(0) == 0 or facts_2.size(0) == 0:
            return (-1, -1)


        h_facts_1 = facts_1[:, 0].unsqueeze(-1)  # shape: (N, 1)
        t_facts_1 = facts_1[:, -1].unsqueeze(-1)   # shape: (N, 1)
        h_facts_2 = facts_2[:, 0].unsqueeze(0) # shape: (1, M)
        t_facts_2 = facts_2[:, -1].unsqueeze(0)  # shape: (1, M)

        correct_matches = torch.sum((h_facts_1 == h_facts_2) & (t_facts_1 == t_facts_2), dim=1)  # shape: (N,)

        if torch.any(correct_matches > 1):
            raise Exception('Duplicate!')

        correct_ratio_facts_1 = torch.sum(correct_matches.float()) / facts_1.size(0)
        correct_ratio_facts_2 = torch.sum(correct_matches.float()) / facts_2.size(0)

        return (correct_ratio_facts_1.item(), correct_ratio_facts_2.item())




    def estimate_rule(self, body: list, target: int) -> tuple:
        facts_specific_relation_in_body = [self.facts[self.facts[:, 1] == relation, :] for relation in body]
        facts_specific_relation_target = self.facts[self.facts[:, 1] == target, :]
        min_facts_num = min(
            [facts.size(0) for facts in facts_specific_relation_in_body] + [facts_specific_relation_target.size(0)])

        if min_facts_num == 0:
            return (-1, -1, 0)
        if len(body) == 1:
            confidence, hc = self.calculate(facts_specific_relation_in_body[0], facts_specific_relation_target)
            return (confidence, hc, facts_specific_relation_in_body[0].size(0))
        elif len(body) == 2:
            hinge_entity = set(facts_specific_relation_in_body[0][:, -1].cpu().numpy()) & set(
                facts_specific_relation_in_body[1][:, 0].cpu().numpy())
            hinge_entity_num = len(hinge_entity)
            if hinge_entity_num <= self.minBodyInstance:
                return (-1, -1, 0)
            facts_body = []
            for hinge in iter(hinge_entity):
                facts_body_first_half = facts_specific_relation_in_body[0][
                                        facts_specific_relation_in_body[0][:, -1] == hinge, :]  #
                facts_body_last_half = facts_specific_relation_in_body[1][
                                       facts_specific_relation_in_body[1][:, 0] == hinge, :]  # 
                for i in range(facts_body_first_half.size(0)):
                    for j in range(facts_body_last_half.size(0)):
                        fact = torch.cat((facts_body_first_half[i][:-1], facts_body_last_half[j][:]), dim=-1)
                        facts_body.append(fact)
            facts_body = torch.stack(facts_body, dim=0).to(facts_specific_relation_target)
            confidence, hc = self.calculate(facts_body, facts_specific_relation_target)
            return (confidence, hc, facts_body.size(0))


        else:
            raise UserWarning('Not yet implement rule mining for body length bigger than 2!')

    def estimate_rule_all_target(self, body: list) -> list:
        torch.cuda.empty_cache()
        facts_specific_relation_in_body = [self.facts[self.facts[:, 1] == relation, :] for relation in body]
        min_facts_num = min([facts.size(0) for facts in facts_specific_relation_in_body])

        if min_facts_num <= self.minBodyInstance:
            return [(-1, -1, 0) for _ in range(len(self.facts_target))]
        if len(body) == 1:
            result_lst = []
            for facts_specific_relation_target in self.facts_target:
                confidence, hc = self.calculate_tensor(facts_specific_relation_in_body[0], facts_specific_relation_target)
                result_lst.append((confidence, hc, facts_specific_relation_in_body[0].size(0)))

            return result_lst
        elif len(body) == 2:
            hinge_entity = set(facts_specific_relation_in_body[0][:, -1].cpu().numpy()) & set(
                facts_specific_relation_in_body[1][:, 0].cpu().numpy())
            hinge_entity_num = len(hinge_entity)
            if hinge_entity_num <= self.minBodyInstance:
                return [(-1, -1, 0) for _ in range(len(self.facts_target))]
            facts_body = []
            for hinge in iter(hinge_entity):
                facts_body_first_half = facts_specific_relation_in_body[0][
                                        facts_specific_relation_in_body[0][:, -1] == hinge, :]  # 
                facts_body_last_half = facts_specific_relation_in_body[1][
                                       facts_specific_relation_in_body[1][:, 0] == hinge, :]  #
                for i in range(facts_body_first_half.size(0)):
                    for j in range(facts_body_last_half.size(0)):
                        fact = torch.cat((facts_body_first_half[i][:-1], facts_body_last_half[j][:]), dim=-1)
                        facts_body.append(fact)
            if len(facts_body) <= self.minBodyInstance:
                return [(-1, -1, 0) for _ in range(len(self.facts_target))]
            facts_body = torch.stack(facts_body, dim=0).to(facts_specific_relation_in_body[0])

            result_lst = []
            for facts_specific_relation_target in self.facts_target:
                confidence, hc = self.calculate_tensor(facts_body, facts_specific_relation_target)
                result_lst.append((confidence, hc, facts_body.size(0)))
            return result_lst

        else:
            raise UserWarning('Not yet implement rule mining for body length bigger than 2!')


    def mine_rule(self):
        for rel1, rel1_id in tqdm(self.rel2id.items()):
            if rel1.lower() != 'na':
                result_lst = self.estimate_rule_all_target([rel1_id])
                for i in range(len(result_lst)):
                    confidence, hc, _ = result_lst[i]
                    if self.facts_target_NameAndID[i][-1] != rel1_id and confidence > self.minC and hc > self.minHC:
                        self.rules[i].body_lst_NAMES.append([self.pid2name(rel1)])
                        self.rules[i].body_lst.append([rel1_id])
                        self.rules[i].confidence_lst.append(confidence)
                        self.rules[i].hc_lst.append(hc)


        for rel1, rel1_id in tqdm(self.rel2id.items()):
            for rel2, rel2_id in self.rel2id.items():
                if rel1.lower() != 'na' and rel2.lower() != 'na' and rel1 != rel2:
                    result_lst = self.estimate_rule_all_target([rel1_id, rel2_id])
                    for i in range(len(result_lst)):
                        confidence, hc, _ = result_lst[i]
                        if confidence > self.minC and hc > self.minHC:
                            self.rules[i].body_lst_NAMES.append([self.pid2name(rel1), self.pid2name(rel2)])
                            self.rules[i].body_lst.append([rel1_id, rel2_id])
                            self.rules[i].confidence_lst.append(confidence)
                            self.rules[i].hc_lst.append(hc)

    def mine_negations(self):
        relation_to_types = {
            'HEAD' : defaultdict(set),
            'TAIL' : defaultdict(set)
        }
        types_to_relations = {
            'HEAD' : defaultdict(set),
            'TAIL' : defaultdict(set)
        }
        for sample in self.data:
            for label in sample['labels']:
                h, r, t = label['h'], label['r'], label['t']
                # r_id = self.rel2id[r]
                h_type, t_type = sample['vertexSet'][h][0]['type'], sample['vertexSet'][t][0]['type']

                relation_to_types['HEAD'][r].add(h_type)
                relation_to_types['TAIL'][r].add(t_type)

                types_to_relations['HEAD'][h_type].add(r)
                types_to_relations['TAIL'][t_type].add(r)
        self.negations =  [Negation(rel_name=rel) for rel, id in self.rel2id.items() if rel.lower() != 'na' and 'anti' not in rel]
        for negation in self.negations:
            negation.build_entity_types(relation_to_types)
            negation.build_negations(types_to_relations=types_to_relations, rel2id=self.rel2id)

def transform_negations(negations, new_rel2id):
    result  = []
    for negation in negations:
        temp = Negation(negation.relation_name)
        temp.head_entity_possible_types = negation.head_entity_possible_types
        temp.tail_entity_possible_types = negation.tail_entity_possible_types

        temp.NEG_tail_entity_AS_TAIL = {new_rel2id[_] for _ in negation.NEG_tail_entity_AS_TAIL}
        temp.NEG_tail_entity_AS_HEAD = {new_rel2id[_] for _ in negation.NEG_tail_entity_AS_HEAD}
        temp.NEG_head_entity_AS_TAIL = {new_rel2id[_] for _ in negation.NEG_head_entity_AS_TAIL}
        temp.NEG_head_entity_AS_HEAD = {new_rel2id[_] for _ in negation.NEG_head_entity_AS_HEAD}
        result.append(temp)
    return result