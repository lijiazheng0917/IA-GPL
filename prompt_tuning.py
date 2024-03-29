import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

import models.graph_prompt as Prompt
from utils.loader import MoleculeDataset
from utils.splitters import scaffold_split, random_scaffold_split, random_split, scaffold_split_abs_value
from models.pretrained_gnn import GNN_graphpred


def train(args, model, device, loader, optimizer, prompt):
    model.train()
    training = True

    criterion = nn.BCEWithLogitsLoss(reduction = "none")
    epoch_loss = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred, c_loss = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt, training)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid) + args.lamda*c_loss
        epoch_loss.append(loss.item())
        loss.backward()

        optimizer.step()
    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    return epoch_loss

def eval(args, model, device, loader, prompt):
    model.eval()
    training = False
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt, training)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    # if len(roc_list) < y_true.shape[1]:
    #     print("Some target is missing!")
    #     print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) 

def main():
    # Load the arguments
    parser = argparse.ArgumentParser(description='PyTorch implementation of instance aware graph prmpt learning')
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1, help='Relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='How the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin", help='GNN gin, gcn, or gat')
    parser.add_argument('--dataset', type=str, default = 'tox21', help='Root directory of dataset. For now, only classification.')
    parser.add_argument('--model_file', type=str, default = '', help='File path to read the model (if there is any)')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "The way of dataset split(e.g., \'scaffold\' for chem data)")   
    parser.add_argument('--eval_train', type=int, default = 1, help='Evaluating training or not')
    parser.add_argument('--num_layers', type=int, default = 1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--num_workers', type=int, default = 4, help='Number of workers for dataset loading')
    parser.add_argument('--warm_up', type=int, default = 20, help='warm up epoch')
    parser.add_argument('--gamma', type=float, default = 1, help='gamma for learning rate scheduler')
    parser.add_argument('--shot_number', type=int, default = 50, help='Number of shots')
    parser.add_argument('--full_few', type=str, default = 'full', help='full or few shot learning')
    parser.add_argument('--hidden_dim', type=int, default = 128, help='hidden dimension for prompt')
    parser.add_argument('--n_codebooks', type=int, default = 20, help='number of codebook vectors')
    parser.add_argument('--n_samples', type=int, default = 10, help='number of samples for each codebook vector')
    parser.add_argument('--temp', type=float, default = 1, help='hyperparameter temp')
    parser.add_argument('--beta', type=float, default = 0.9, help='hyperparameter beta')
    parser.add_argument('--lamda', type=float, default = 0.01, help='hyperparameter lamda')
    
    args = parser.parse_args()

    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    wandb.init(project='IA-GPL',
               name=args.dataset + '_' + args.full_few,
               config=args,
               )
    
    # Run for 5 times,
    result_list = []
    for i in range(5): 

        # seed everything
        runseed = args.runseed + i
        torch.manual_seed(runseed)
        np.random.seed(runseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(runseed)

        #Bunch of classification tasks
        if args.dataset == "tox21":
            num_tasks = 12
        elif args.dataset == "hiv":
            num_tasks = 1
        elif args.dataset == "pcba":
            num_tasks = 128
        elif args.dataset == "muv":
            num_tasks = 17
        elif args.dataset == "bace":
            num_tasks = 1
        elif args.dataset == "bbbp":
            num_tasks = 1
        elif args.dataset == "toxcast":
            num_tasks = 617
        elif args.dataset == "sider":
            num_tasks = 27
        elif args.dataset == "clintox":
            num_tasks = 2
        else:
            raise ValueError("Invalid dataset name.")

        #set up dataset
        dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
        print(dataset)

        if args.split == "scaffold":
            if args.full_few == 'full':
                smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
                train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
                print("scaffold splitting for full shot learning")
            elif args.full_few == 'few':
                smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
                train_dataset, valid_dataset, test_dataset = scaffold_split_abs_value(dataset, smiles_list, null_value=0,
                                                                                    number_train=args.shot_number, 
                                                                                    frac_valid=0.1, frac_test=0.1)
                print("scaffold splitting for few shot learning")
            else:
                raise ValueError("Invalid split option.")
        elif args.split == "random":
            if args.full_few == 'full':
                train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
                print("random splitting for full shot learning")
            elif args.full_few == 'few':
                train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
                train_dataset = train_dataset[:args.shot_number]
                print("random splitting for few shot learning")
        else:
            raise ValueError("Invalid split option.")

        # set up data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        # set up the model
        model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, head_layer = args.num_layers)
        if not args.model_file == "":
            model.from_pretrained(args.model_file)
        print(model)
        model.to(device)

        # set up instance aware prompts
        prompt = Prompt.PromptVQ(in_channels=args.emb_dim, 
                                hidden_channels=args.hidden_dim,
                                n_codebooks=args.n_codebooks,
                                n_samples=args.n_samples,
                                temp=args.temp,
                                beta=args.beta).to(device)

        #set up optimizer
        model_param_group = []
        model_param_group.append({"params": prompt.parameters(), 'lr':args.lr})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        optimizer = optim.Adam(model_param_group, weight_decay=args.decay, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)
        print(optimizer)
        
        train_acc_list = []
        val_acc_list = []
        best_acc = 0

        for epoch in tqdm(range(args.epochs)): 
            train_loss = train(args, model, device, train_loader, optimizer, prompt)
            scheduler.step()

            # evaluate on training set
            if args.eval_train: 
                train_acc = eval(args, model, device, train_loader, prompt)
            else:
                train_acc = 0
            
            # evaluate on validation set
            val_acc = eval(args, model, device, val_loader, prompt)

            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})

            # save the best model
            if val_acc > best_acc and epoch > args.warm_up:
                patience = 0
                best_acc = val_acc
                torch.save(prompt.state_dict(), './ckpt/best_prompt_{}_gpf.pt'.format(args.dataset))
                torch.save(model.state_dict(), './ckpt/best_model.pt')
                print("best model saved at epoch:", epoch)
                best_epoch = epoch
            # if use early stop strategy
            # else:
            #     patience += 1
            #     if patience == args.early_stop:
            #         print('early stop at epoch:', epoch)
            #         break

            val_acc_list.append(val_acc)
            train_acc_list.append(train_acc)

        # final test result
        model.load_state_dict(torch.load('./ckpt/best_model.pt'))
        prompt.load_state_dict(torch.load('./ckpt/best_prompt_{}_gpf.pt'.format(args.dataset)))
        test_acc = eval(args, model, device, test_loader, prompt)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
        result_list.append(test_acc)
        print('best_epoch:',best_epoch)

    # compute the mean and std of the results
    m, std = np.mean(result_list), np.std(result_list)
    with open('results.log', 'a+') as f:
        f.write(args.dataset + ' '  + str(m) + ' ' + str(std) + '\n')

if __name__ == "__main__":
    main()