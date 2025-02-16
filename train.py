import argparse
import os

import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel, Logical_Pseudo_Labeling_Module
from utils import set_seed, collate_fn, _update_mean_model_variables
from prepro import read_docred, read_dwie
from evaluation import to_official, official_evaluate, official_evaluate_benchmark
import wandb
from tqdm import tqdm
import re
global dataset

def train(args, teacher_model, student_model, train_features, dev_features, test_features):
    def finetune(features, teacher_model, student_model, num_pretrain_epoch, num_epoch, num_steps):

        new_layer = ["extractor", "bilinear", "classifier"]
        optimizer_grouped_parameters_TEACHER = [
            {"params": [p for n, p in teacher_model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in teacher_model.named_parameters() if any(nd in n for nd in new_layer)],
             "lr": 1e-4},
        ]
        optimizer_TEACHER = AdamW(optimizer_grouped_parameters_TEACHER, lr=args.learning_rate, eps=args.adam_epsilon)
        teacher_model, optimizer_TEACHER = amp.initialize(teacher_model, optimizer_TEACHER, opt_level="O1", verbosity=0)
        teacher_model.zero_grad()
        # ==============================Pre-Training START===============================
        best_score = -1
        if args.load_teacher_path == "":
            train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
            train_iterator = range(int(num_pretrain_epoch))
            total_steps = int(len(train_dataloader) * num_pretrain_epoch // args.gradient_accumulation_steps)
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer_TEACHER, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            print("Total Pretraining steps: {}".format(total_steps))
            print("Warmup Pretraining steps: {}".format(warmup_steps))
            for epoch in train_iterator:
                teacher_model.zero_grad()
                for step, batch in enumerate(tqdm(train_dataloader, desc=f"Pre-Training epoch: {epoch}", mininterval=30)):
                    teacher_model.train()
                    inputs = {'input_ids': batch[0].to(args.device),
                              'attention_mask': batch[1].to(args.device),
                              'labels': batch[2],
                              'entity_pos': batch[3],
                              'hts': batch[4],
                              'epoch':epoch
                              }
                    outputs = teacher_model(**inputs)
                    loss = outputs[0] / args.gradient_accumulation_steps
                    # Update GAPs_bank
                    teacher_model.logical_pseudo_labeling_module.update_bank(outputs[-1])
                    #
                    with amp.scale_loss(loss, optimizer_TEACHER) as scaled_loss:
                        scaled_loss.backward()
                    if step % args.gradient_accumulation_steps == 0:
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_TEACHER), args.max_grad_norm)
                        optimizer_TEACHER.step()
                        scheduler.step()
                        teacher_model.zero_grad()
                        num_steps += 1

                    wandb.log({"Pre-Train loss": loss.item()}, step=num_steps)
                    if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                        dev_score, dev_output, dev_loss = evaluate(args, teacher_model, dev_features, tag="dev")
                        wandb.log( {"Teacher " + key: value for key, value in dev_output.items()}, step=num_steps)
                        wandb.log({"Teacher Dev Loss": dev_loss}, step=num_steps)
                        print(dev_output)
                        if dev_score > best_score:
                            best_score = dev_score
                            pred = report(args, teacher_model, test_features)
                            with open("result.json", "w") as fh:
                                json.dump(pred, fh)
                            if args.teacher_save_path != "":
                                torch.save(teacher_model.state_dict(), args.teacher_save_path)
            del train_dataloader, train_iterator, scheduler, optimizer_TEACHER, optimizer_grouped_parameters_TEACHER
        else:
            teacher_model.load_state_dict(torch.load(args.load_teacher_path))
        # ==============================Pre-Training END===============================
        # ==============================Training START=================================
        teacher_model.eval() # not producing gradient
        optimizer_grouped_parameters_STUDENT = [
            {"params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in new_layer)],
             "lr": 1e-4},
        ]
        optimizer_STUDENT = AdamW(optimizer_grouped_parameters_STUDENT, lr=args.learning_rate, eps=args.adam_epsilon)
        student_model, optimizer_STUDENT = amp.initialize(student_model, optimizer_STUDENT, opt_level="O1", verbosity=0)
        # NOTE：copy teacher's parameter to student
        student_model.load_state_dict(teacher_model.state_dict())
        student_model.zero_grad()
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer_STUDENT, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total Training steps: {}".format(total_steps))
        print("Warmup Training steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            # print GAPs_bank
            teacher_model.logical_pseudo_labeling_module.print_bank()
            #
            student_model.zero_grad()
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training epoch: {epoch}", mininterval=30)):
                student_model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'epoch':epoch
                          }
                with torch.no_grad():
                    pseudo_labels, instance_mask, logits = teacher_model.pseudo_label(**inputs)
                    wandb.log({'Pseudo Ratio' : instance_mask.sum().item() / torch.ones_like(instance_mask).sum().item()}, step=num_steps)
                    wandb.log({'Positive Ratio' : (pseudo_labels[:, 0].sum() / pseudo_labels[:, 0].shape[0]).item() }, step=num_steps)

                    # ------Update GAPs_bank------
                    teacher_model.logical_pseudo_labeling_module.update_bank(logits)
                    #------------------------------
                inputs['labels'], inputs['instance_mask'] = pseudo_labels, instance_mask
                # student!!
                outputs = student_model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps

                with amp.scale_loss(loss, optimizer_STUDENT) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_STUDENT), args.max_grad_norm)
                    optimizer_STUDENT.step()
                    scheduler.step()
                    student_model.zero_grad()
                    num_steps += 1

                    # ------teacher EMA update START---------
                    _update_mean_model_variables(student_model, teacher_model, args.EMA_lambda)
                    # ------teacher EMA update END-----------
                wandb.log({"Train loss": loss.item()}, step=num_steps)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    # evaluate teacher
                    dev_score, dev_output, dev_loss = evaluate(args, teacher_model, dev_features, tag="dev")
                    wandb.log({"Teacher " + key: value for key, value in dev_output.items()}, step=num_steps)
                    wandb.log({"Teacher Dev Loss": dev_loss}, step=num_steps)
                    print('Teacher:', dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, teacher_model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.teacher_save_path != "":
                            torch.save(teacher_model.state_dict(), args.teacher_save_path)

                    # evaluate student
                    dev_score, dev_output, dev_loss = evaluate(args, student_model, dev_features, tag="dev")
                    wandb.log({"Student " + key: value for key, value in dev_output.items()}, step=num_steps)
                    wandb.log({"Student Dev Loss": dev_loss}, step=num_steps)
                    print('Student:', dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, student_model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.student_save_path != "":
                            torch.save(student_model.state_dict(), args.student_save_path)
        # ==============================Training END===============================
        return num_steps

    num_steps = 0
    set_seed(args)

    finetune(train_features, teacher_model, student_model, args.num_pretrain_epochs, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev", benchmark=False):
    global dataset
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    losses = []

    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  #'epoch':-1
                  }
        with torch.no_grad():
            loss, pred, logits = model(**inputs)
            losses.append(loss.item())
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    dev_loss = np.mean(losses)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features, dataset)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, args, tag=tag)
    else:
        best_f1, best_f1_ign = 0, 0
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }


    if benchmark and dataset == 'docred':
        official_evaluate_benchmark(ans, args.data_dir, args, tag=tag)

    return best_f1, output, dev_loss


def report(args, model, features):
    global dataset

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features, dataset)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--part_train_file", default="./dataset/docred/train_annotated.json", type=str)

    parser.add_argument("--dev_file", default="./dataset/docred/dev_revised.json", type=str)
    parser.add_argument("--test_file", default="./dataset/docred/test_revised.json", type=str)

    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="The number of workers")
    parser.add_argument("--num_pretrain_epochs", default=5.0, type=float,
                        help="Total number of pre-training epochs to perform.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--WANDB_MODE", type=str , default="disabled")
    parser.add_argument("--loss_fn", type=str , default="NCRL",
                        help="Number of relation types in dataset.")
    parser.add_argument("--NCRL_REG", type=int , default=0,
                        help="Whther to use margin regularization.")
    parser.add_argument("--NS", action="store_true") # use negative sampling as a baseline

    # ========================================================================
    # Teacher-Stuent Framework
    parser.add_argument("--negative_sampling_ratio", default=1.0, type=float)
    parser.add_argument("--EMA_lambda", default=0.999, type=float)
    parser.add_argument("--ratio_pos", default=1.0, type=float)
    parser.add_argument("--ratio_neg", default=1.0, type=float)
    parser.add_argument("--load_teacher_path", default="", type=str)
    # ========================================================================
    # Logical Diagnosis
    parser.add_argument("--diagnose_mode", default="vanilla", choices=["vanilla", "rule"] ,type=str)
    parser.add_argument("--sampling_mode", default="uniform", choices=["uniform", "best", 'worst'], type=str)
    parser.add_argument("--indicator", default="pos", choices=["pos", "min", 'prob'] ,type=str)

    parser.add_argument("--rule_path", default="./mined_rules/rule_docred.pl", type=str)
    parser.add_argument("--minC", default=0.85, type=float)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--gamma", default=2.0, type=float)
    parser.add_argument("--top_K", type=int , default=25)

    # ========================================================================
    parser.add_argument("--norm_by_sample", default=False, action="store_true") # norm by sample / norm by classes
    parser.add_argument("--device", default=0, type=int)


    args = parser.parse_args()
    os.environ["WANDB_MODE"] = args.WANDB_MODE
    wandb.init(project="LogicST")

    # ---------------------
    print('='*50)
    print('transformer_type:', args.transformer_type)
    print('seed:', args.seed)
    print('diagnose_mode:', args.diagnose_mode)
    print('rule_path:', args.rule_path)
    print('sampling_mode:', args.sampling_mode)
    print('indicator:', args.indicator)

    print('pre-training epochs:', args.num_pretrain_epochs)
    print('negative_sampling_ratio:', args.negative_sampling_ratio)
    print('EMA_lambda:', args.EMA_lambda)
    print('ratio_pos:', args.ratio_pos)
    print('ratio_neg:', args.ratio_neg)
    print('min Confidence of rules:', args.minC)
    print('eta in GAPs bank:', args.eta)
    print('gamma in diagnose:', args.gamma)
    print('learning_rate:', args.learning_rate)
    print('='*50)
    # ==================
    global dataset
    if 'docred' in args.data_dir:
        dataset = 'docred'
    elif 'dwie' in args.data_dir:
        dataset = 'dwie'
    else:
        raise Exception('ERROR')

    pattern = r'\d+\.\d+'
    matches = re.findall(pattern, args.part_train_file)
    if len(matches) > 0:
        SPECIFIC_dataset_NAME = matches[0]
    else:
        SPECIFIC_dataset_NAME = ""

    save_prefix = f"trained_model/{dataset}{SPECIFIC_dataset_NAME}_{args.transformer_type}-{args.loss_fn}"
    for name, condition in [
        (f"_{args.diagnose_mode}", args.num_train_epochs > 0),
        (f'_Reg', bool(args.NCRL_REG) and args.loss_fn == "NCRL"),
        (f'_ext', 'ext' in args.part_train_file),
        (f'_eta{args.eta}', args.diagnose_mode == 'rule' and args.eta != 0 and args.indicator == 'prob' and args.num_train_epochs > 0),
        (f'_gamma{args.gamma}', args.diagnose_mode == 'rule' and args.gamma  > 1 and args.indicator == 'prob' and args.num_train_epochs > 0)

    ]:
        if condition:
            save_prefix += name

    if args.NS:
        print(f'Only use negative sampling ratio of {args.negative_sampling_ratio}')
        save_prefix = f"trained_model/{dataset}{SPECIFIC_dataset_NAME}_{args.transformer_type}-{args.loss_fn}-NS{args.negative_sampling_ratio}"

        args.num_train_epochs = 0
        assert args.load_teacher_path == ""


    args.teacher_save_path = save_prefix + f"-Teacher-best-{args.seed}.pt"
    args.student_save_path = save_prefix + f"-Student-best-{args.seed}.pt"

    print('teacher_save_path: ', args.teacher_save_path )
    print('studnet_save_path: ', args.student_save_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read_map = {
        'dwie': read_dwie,
        'docred': read_docred
    }
    read = read_map[dataset]

    if args.transformer_type == 'bert':
        prefix = ""
    elif args.transformer_type == 'roberta':
        prefix = '_roberta_'
    else:
        raise Exception(args.transformer_type)
    print('save tokenizer prefix:', prefix)

    if not os.path.exists(args.part_train_file + f'{prefix}.pt'):
        part_train_features = read(args.part_train_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(part_train_features, args.part_train_file + f'{prefix}.pt')
    else:
        part_train_features = torch.load(args.part_train_file + f'{prefix}.pt')

    if not os.path.exists(args.dev_file + f'{prefix}.pt'):
        dev_features = read(args.dev_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(dev_features, args.dev_file + f'{prefix}.pt')
    else:
        dev_features = torch.load(args.dev_file + f'{prefix}.pt')

    if not os.path.exists(args.test_file + f'{prefix}.pt'):
        test_features = read(args.test_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(test_features, args.test_file + f'{prefix}.pt')
    else:
        test_features = torch.load(args.test_file + f'{prefix}.pt')

    model1 = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    model2 = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    config.diagnose_mode, config.sampling_mode, config.num_labels = args.diagnose_mode, args.sampling_mode, args.num_class
    config.negative_sampling_ratio, config.norm_by_sample = args.negative_sampling_ratio, args.norm_by_sample
    config.ratio_pos, config.ratio_neg , config.EMA_lambda = args.ratio_pos, args.ratio_neg, args.EMA_lambda
    config.device = args.device

    set_seed(args)
    teacher_model = DocREModel(config, model1, num_labels=args.num_labels, args=args)
    teacher_model.to(args.device)

    student_model = DocREModel(config, model2, num_labels=args.num_labels, args=args)
    student_model.to(args.device)

    if 'docred' in args.data_dir:
        rel2id = json.load(open('./dataset/docred/rel2id.json', 'r'))
        rel_info = json.load(open('./dataset/docred/rel_info.json', 'r'))
    elif 'dwie' in args.data_dir:
        rel2id = json.load(open('./dataset/dwie/meta/rel2id.json', 'r'))
        rel_info = None
    else:
        raise Exception('Not implemented!')
    teacher_model.logical_pseudo_labeling_module = Logical_Pseudo_Labeling_Module(config, rule_path=args.rule_path, minC=args.minC, rel2id=rel2id, rel_info=rel_info,
                                                                                  indicator=args.indicator, top_K=args.top_K, device=device, eta=args.eta, gamma=args.gamma, dataset=dataset)

    if args.load_path == "":  # Training
        train(args, teacher_model, student_model , part_train_features, dev_features, test_features)

        model = amp.initialize(teacher_model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.teacher_save_path))
        dev_score, dev_output, dev_loss = evaluate(args, model, dev_features, tag="dev", benchmark=True)
        print(dev_output)
        test_score, test_output, test_loss = evaluate(args, model, test_features, tag="test", benchmark=True)
        print(test_output)

    else:  # Testing
        print(f"Load from {args.load_path}")
        model = amp.initialize(teacher_model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output, dev_loss = evaluate(args, model, dev_features, tag="dev", benchmark=True)
        print(dev_output)
        test_score, test_output, test_loss = evaluate(args, model, test_features, tag="test", benchmark=True)
        print(test_output)
        pred = report(args, model, test_features)
        with open(f"{args.transformer_type}_result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
