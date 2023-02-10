import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data import create_loaders
from image_classification_model import ImageClassificationModel, find_last_checkpoint_file
from logger import Logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_curve
from upload_s3 import multiup
import seaborn as sns
import random

class2label = {'not_cancer':0, 'cancer':1}

class SigmoidFocalLoss(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    def __init__(self, alpha=.25, gamma=2, reduction="none"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        #print("p_t: ", p_t)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            #print("alpha_t: ", alpha_t)
            loss = alpha_t * loss
            #print("loss: ", loss)

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


def pfbeta(labels, predictions, beta=1., epsilon=1e-07):
    "label e prediction ar numpy array (dim,)"
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def train_batch(inputs, labels, model, optimizer, criterion):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels.unsqueeze(1).float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


@torch.no_grad()
def find_list_label_pred_propba(inputs, labels, model):
    model.eval()
    outputs = model(inputs)
    labels = labels.cpu().numpy().tolist()
    pred_proba = model.post_act_sigmoid(outputs)
    pred_proba = pred_proba.squeeze().cpu().numpy().tolist()
    return labels, pred_proba


@torch.no_grad()
def val_loss(inputs, labels, model, criterion):
    model.eval()
    outputs = model(inputs)
    val_loss = criterion(outputs, labels.unsqueeze(1).float())
    return val_loss.item()


def train_model(device,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                classification_dataloader_train,
                classification_dataloader_val,
                best_epoch,
                num_epoch,
                best_test_pfs,
                checkpoint_dir,
                saving_dir_experiments,
                logger,
                epoch_start_unfreeze=None,
                layer_start_unfreeze=None,
                aws_bucket=None,
                aws_directory=None,
                scheduler_type=None):
    train_losses, train_pfs = [], []
    val_losses, val_pfs = [], []

    print("Start training")
    freezed = True
    for epoch in range(best_epoch, num_epoch):
        logger.log(f'Epoch {epoch}/{num_epoch - 1}')

        if epoch_start_unfreeze is not None and epoch >= epoch_start_unfreeze and freezed:
            print("****************************************")
            print("unfreeze the base model weights")

            if layer_start_unfreeze is not None:
                print("unfreeze the layers greater and equal to layer_start_unfreeze: ", layer_start_unfreeze)
                # in this case unfreeze only the layers greater and equal the unfreezing_block layer
                for i, properties in enumerate(model.named_parameters()):
                    if i >= layer_start_unfreeze:
                        # print("Unfreeze model layer: {} -  name: {}".format(i, properties[0]))
                        properties[1].requires_grad = True
            else:
                # in this case unfreeze all the layers of the model
                print("unfreeze all the layer of the model")
                for name, param in model.named_parameters():
                    param.requires_grad = True

            freezed = False

            for name, param in model.named_parameters():
                print("Layer name: {} - requires_grad: {}".format(name, param.requires_grad))
            print("****************************************")

        # define empty lists for the values of the loss then at the end I take the average and I get the final values of the whole era
        train_epoch_losses, train_epoch_labels, train_epoch_preds_proba = [], [], []
        val_epoch_losses, val_epoch_labels, val_epoch_preds_proba = [], [], []

        # cycle on all train batches of the current epoch by executing the train_batch function
        for inputs, labels in tqdm(classification_dataloader_train, desc=f"epoch {str(epoch)} | train"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_loss = train_batch(inputs, labels, model, optimizer, criterion)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        # cycle on all train batches of the current epoch by calculating their accuracy
        for inputs, labels in tqdm(classification_dataloader_train, desc=f"epoch {str(epoch)} | train"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_array, pred_proba_array = find_list_label_pred_propba(inputs, labels, model)
            # extend labels_array
            train_epoch_labels.extend(labels_array)
            # extend pred_proba_array
            train_epoch_preds_proba.extend(pred_proba_array)

        # evaluate train_epoch_auc
        train_epoch_labels = np.array(train_epoch_labels)
        train_epoch_preds_proba = np.array(train_epoch_preds_proba)
        precision_proba, recall_proba, _ = precision_recall_curve(train_epoch_labels, train_epoch_preds_proba)
        # print("train_epoch_labels.shape", train_epoch_labels.shape)
        # print("train_epoch_preds_proba.shape", train_epoch_preds_proba.shape)
        # print("precision_proba.shape", precision_proba.shape)
        # print("recall_proba.shape", recall_proba.shape)
        train_epoch_auc = metrics.auc(recall_proba, precision_proba)

        # evaluate probabilistic fscore
        train_epoch_pfs = pfbeta(train_epoch_labels, train_epoch_preds_proba)

        # cycle on all batches of val of the current epoch by calculating the loss function and auc
        for inputs, labels in tqdm(classification_dataloader_val, desc=f"epoch {str(epoch)} | val"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            validation_loss = val_loss(inputs, labels, model, criterion)
            val_epoch_losses.append(validation_loss)
            labels_array, pred_proba_array = find_list_label_pred_propba(inputs, labels, model)
            # extend labels_array
            val_epoch_labels.extend(labels_array)
            # extend pred_proba_array
            val_epoch_preds_proba.extend(pred_proba_array)

        # evaluate val_epoch_auc
        val_epoch_labels = np.array(val_epoch_labels)
        val_epoch_preds_proba = np.array(val_epoch_preds_proba)
        precision_proba, recall_proba, _ = precision_recall_curve(val_epoch_labels, val_epoch_preds_proba)
        val_epoch_auc = metrics.auc(recall_proba, precision_proba)
        # print("val_epoch_labels.shape", val_epoch_labels.shape)
        # print("val_epoch_preds_proba.shape", val_epoch_preds_proba.shape)
        # print("precision_proba.shape", precision_proba.shape)
        # print("recall_proba.shape", recall_proba.shape)
        val_epoch_loss = np.mean(val_epoch_losses)

        # evaluate probabilistic fscore
        val_epoch_pfs = pfbeta(val_epoch_labels, val_epoch_preds_proba)

        phase = 'train'
        # logger.log(f'{phase} LR: {optimizer.param_groups[0]['lr']:.9f} - Loss: {train_epoch_loss:.4f} - AUC: {train_epoch_auc:.4f}')
        phase = 'val'
        # logger.log(f'{phase} LR: {optimizer.param_groups[0]['lr']:.9f} - Loss: {val_epoch_loss:.4f} - AUC: {val_epoch_auc:.4f}')
        print(
            "Epoch: {} - LR:{} - Train Loss: {:.4f} - Train AUC: {:.4f} - Train PFS: {:.4f} - Val Loss: {:.4f} - Val AUC: {:.4f} - Val PFS: {:.4f}".format(
                int(epoch), optimizer.param_groups[0]['lr'], train_epoch_loss, train_epoch_auc, train_epoch_pfs,
                val_epoch_loss, val_epoch_auc, val_epoch_pfs))
        logger.log("-----------")

        train_losses.append(train_epoch_loss)
        train_pfs.append(train_epoch_pfs)
        val_losses.append(val_epoch_loss)
        val_pfs.append(val_epoch_pfs)

        print("Plot learning curves")
        plot_learning_curves(epoch - best_epoch + 1, train_losses, val_losses, train_pfs, val_pfs, checkpoint_dir)

        if best_test_pfs < val_epoch_pfs:
            print("We have a new best model! Save the model")

            # update best_test_pfs
            best_test_pfs = val_epoch_pfs
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_test_pfs': best_test_pfs
            }
            print("Save best checkpoint at: ", os.path.join(checkpoint_dir, 'best.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'best.pth'), _use_new_zipfile_serialization=False)
            print("Save latest checkpoint at: ", os.path.join(checkpoint_dir, 'latest.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'latest.pth'), _use_new_zipfile_serialization=False)

        else:
            print("Save the current model")

            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_test_pfs': best_test_pfs
            }
            print("Save latest checkpoint at: ", os.path.join(checkpoint_dir, 'latest.pth'))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'latest.pth'), _use_new_zipfile_serialization=False)

        if scheduler_type == "ReduceLROnPlateau":
            lr_scheduler.step(val_epoch_pfs)
        else:
            lr_scheduler.step()
        torch.cuda.empty_cache()

        if aws_bucket is not None and aws_directory is not None:
            print('Upload on S3')
            multiup(aws_bucket, aws_directory, saving_dir_experiments)
        print("---------------------------------------------------------")

    print("End training")
    return


def test_model(device,
               model,
               classification_dataloader,
               path_save,
               type_dataset):
    y_test_true = []
    y_test_predicted = []
    y_test_preds_proba = []
    total = 0
    model = model.eval()

    with torch.no_grad():
        # cycle on all train batches of the current epoch by calculating their accuracy
        for inputs, labels in classification_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # Get the predicted classes
            pred_proba = model.post_act_sigmoid(outputs)
            # transform to numpy
            labels = labels.cpu().numpy()
            pred_proba = pred_proba.squeeze().cpu().numpy()
            preds_labels = np.round(pred_proba)  # 0 and 1 -> con soglia al 0.5
            # extend lists
            y_test_true.extend(labels.tolist())
            y_test_predicted.extend(preds_labels.tolist())
            y_test_preds_proba.extend(pred_proba.tolist())
            numero_immagini = len(labels.tolist())
            total += numero_immagini

        # report predictions and true values to numpy array
        print('Number of tested images: ', total)
        y_test_true = np.array(y_test_true)
        y_test_predicted = np.array(y_test_predicted)
        y_test_preds_proba = np.array(y_test_preds_proba)
        print('y_test_true.shape: ', y_test_true.shape)
        print('y_test_predicted.shape: ', y_test_predicted.shape)

        print('Accuracy: ', accuracy_score(y_test_true, y_test_predicted))
        print(metrics.classification_report(y_test_true, y_test_predicted))

        ## Plot confusion matrix -> con soglia al 0.5
        cm = metrics.confusion_matrix(y_test_true, y_test_predicted)
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=class2label.keys(),
               yticklabels=class2label.keys(), title="Confusion matrix")
        plt.yticks(rotation=0)
        fig.savefig(os.path.join(path_save, type_dataset + "_confusion_matrix.png"))

        # evaluate val_epoch_auc
        precision_proba, recall_proba, thresholds = precision_recall_curve(y_test_true, y_test_preds_proba)
        auc = metrics.auc(recall_proba, precision_proba)
        print("Precision-Recall AUC: ", auc)

        # evaluate probabilistic fscore
        pfs = pfbeta(y_test_true, y_test_preds_proba)
        print("Probabilistic F1-score: ", pfs)

        # Plot precision-recall curve -> con soglia al 0.5
        auc_pr = metrics.auc(recall_proba, precision_proba)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(recall_proba, precision_proba, linewidth=2, label='AUC = %0.2f' % (auc_pr))
        # plot the current threshold on the line
        close_default_clf = np.argmin(np.abs(thresholds - 0.5))
        ax.plot(recall_proba[close_default_clf], precision_proba[close_default_clf], '^', c='k', markersize=10)
        fig.savefig(os.path.join(path_save, type_dataset + "_precisione-recall-curve.png"))

        ## Save report in a txt
        target_names = list(class2label.keys())
        cr = metrics.classification_report(y_test_true, y_test_predicted, target_names=target_names)
        f = open(os.path.join(path_save, type_dataset + "_report.txt"), 'w')
        f.write('Title\n\nClassification Report\n\n{}'.format(cr))
        f.write('\n')
        f.write('\n')
        f.write('Probabilistic F1-score: ' + str(pfs))
        f.write('\n')
        f.write('\n')
        f.write('Precision-Recall AUC: ' + str(auc))
        f.close()


def plot_learning_curves(epochs, train_losses, val_losses, train_auc, val_auc, path_save):
    '''
    La funzione plotta le learning curves sul train e validation set di modelli già allenati
    '''
    x_axis = range(0, epochs)

    plt.figure(figsize=(27, 9))
    plt.suptitle('Learning curves ', fontsize=18)
    # primo plot
    plt.subplot(121)
    plt.plot(x_axis, train_losses, label='Training Loss')
    plt.plot(x_axis, val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Train and Validation Losses', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)

    # secondo plot
    plt.subplot(122)
    plt.plot(x_axis, train_auc, label='Training AUC')
    plt.plot(x_axis, val_auc, label='Validation AUC')
    plt.legend()
    plt.title('Train and Validation accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.savefig(os.path.join(path_save, "learning_curves.png"))


def run_train_test_model(cfg, do_train, do_test, aws_bucket=None, aws_directory=None):

    seed_everything(42)
    checkpoint = None
    best_epoch = 0
    best_test_pfs = 0

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    dataset_path = dataset_cfg['dataset_path']
    path_dataset_train_csv = dataset_cfg['path_dataset_train_csv']
    path_dataset_val_csv = dataset_cfg['path_dataset_val_csv']
    path_dataset_test_csv = dataset_cfg.get("path_dataset_test_csv", None)

    saving_dir_experiments = model_cfg['saving_dir_experiments']
    saving_dir_model = model_cfg['saving_dir_model']
    num_epoch = model_cfg['num_epoch']
    epoch_start_unfreeze = model_cfg.get("epoch_start_unfreeze", None)
    layer_start_unfreeze = model_cfg.get("layer_start_unfreeze", None)
    batch_size = dataset_cfg['batch_size']
    criterion_type = model_cfg['criterion_type']
    pos_class_weight = model_cfg.get("pos_class_weight", None)
    alpha = model_cfg.get("alpha", None)
    scheduler_type = model_cfg['scheduler_type']
    learning_rate = model_cfg['learning_rate']
    lr_patience = model_cfg.get("lr_patience", None)
    scheduler_step_size = model_cfg.get("scheduler_step_size", None)
    lr_factor = model_cfg.get("lr_factor", None)
    T_max = model_cfg.get("T_max", None)
    eta_min = model_cfg.get("eta_min", None)


    # 2 - load csv dataset
    df_dataset_train = pd.read_csv(path_dataset_train_csv)
    df_dataset_val = pd.read_csv(path_dataset_val_csv)
    df_dataset_test = None
    if path_dataset_test_csv is not None:
        df_dataset_test = pd.read_csv(path_dataset_test_csv)

    # 3 -  create the directories with the structure required by the project
    print("Create the project structure")
    print("saving_dir_experiments: ", saving_dir_experiments)
    saving_dir_model = os.path.join(saving_dir_experiments, saving_dir_model)
    print("saving_dir_model: ", saving_dir_model)
    os.makedirs(saving_dir_experiments, exist_ok=True)
    os.makedirs(saving_dir_model, exist_ok=True)

    # 3 - load log configuration
    logger = Logger(exp_path=saving_dir_model)

    # 4 - create the dataloaders
    train_loader, test_loader, _ = create_loaders(df_dataset_train=df_dataset_train,
                                                  df_dataset_val=df_dataset_val,
                                                  df_dataset_test=None,
                                                  cfg=cfg,
                                                  dataset_path=dataset_path,
                                                  batch_size=batch_size)

    # 5 - set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # 6 - download the model
    model = ImageClassificationModel(cfg).to(device)
    for i, properties in enumerate(model.named_parameters()):
        print("Model layer: {} -  name: {} - requires_grad: {} ".format(i, properties[0], properties[1].requires_grad))

    checkpoint_dir = saving_dir_model

    if do_train:
        # look if exist a checkpoint
        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir)
        if path_last_checkpoint is not None:
            print("Load checkpoint from path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)

        # Set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # set the scheduler
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode='max',
                                          patience=lr_patience,
                                          verbose=True,
                                          factor=lr_factor)
        elif scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=scheduler_step_size,
                                                        gamma=lr_factor)
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                   T_max=T_max,
                                                                   eta_min=eta_min)

        # set the loss
        if criterion_type == "BCE_LOSS":
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_class_weight))
        elif criterion_type == "FOCAL_LOSS":
            criterion = SigmoidFocalLoss(alpha=alpha, reduction="mean")

        if checkpoint is not None:
            print('Load the optimizer from the last checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint["scheduler"])

            print('Latest epoch of the checkpoint: ', checkpoint['epoch'])
            print('Setting the new starting epoch: ', checkpoint['epoch'] + 1)
            best_epoch = checkpoint['epoch'] + 1

            print('Setting best fps from checkpoint: ', checkpoint['best_test_pfs'])
            best_test_pfs = checkpoint['best_test_pfs']

        # run train model
        train_model(device=device,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    classification_dataloader_train=train_loader,
                    classification_dataloader_val=test_loader,
                    best_epoch=best_epoch,
                    num_epoch=num_epoch,
                    best_test_pfs=best_test_pfs,
                    checkpoint_dir=checkpoint_dir,
                    saving_dir_experiments=saving_dir_experiments,
                    logger=logger,
                    epoch_start_unfreeze=epoch_start_unfreeze,
                    layer_start_unfreeze=layer_start_unfreeze,
                    scheduler_type=scheduler_type)
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

    if do_test:

        print("Train and Val Dataset with best checkpoint")

        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir=checkpoint_dir, use_best_checkpoint=True)
        if path_last_checkpoint is not None:
            print("Upload the best checkpoint at the path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)
        '''
        # execute the inferences on the train, val and test set
        print("Inference on train dataset")
        test_model(device=device,
                   model=model,
                   classification_dataloader=train_loader,
                   path_save=checkpoint_dir,
                   type_dataset="train")

        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")
        '''
        print("Inference on val dataset")
        test_model(device=device,
                   model=model,
                   classification_dataloader=test_loader,
                   path_save=checkpoint_dir,
                   type_dataset="val")

        if aws_bucket is not None and aws_directory is not None:
            print("Final upload on S3")
            multiup(aws_bucket, aws_directory, saving_dir_experiments)

        print("End test")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True