import os
import shap
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.metrics import roc_auc_score


class TrainerModel:
    def __init__(self, model, dataset, exp_name, type_model="NN", lr=3e-4, weight_decay=0.005):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.type_model = type_model
        if self.type_model == "Boosting":
            self.model = model
        else:
            self.model = model.to(self.device)
        
        self.dataset = dataset

        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs", f"{exp_name}.csv")
        self.writer = SummaryWriter(log_dir=f'tensorboard/{exp_name}')

        if type_model != "Boosting":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            self.loss_fn = binary_cross_entropy_with_logits
            self.n_epoch = None
            self.batch_size = None
        else:
            self.catboost_params = {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'early_stopping_rounds': 50,
                'verbose': False
            }

        self.metrics_history = {
            'train_loss': [], 'test_loss': [], 'val_loss': [],
            'train_roc_auc': [], 'test_roc_auc': [], 'val_roc_auc': [],
            'train_gini': [], 'test_gini': [], 'val_gini': []
            # 'train_recall': [], 'val_recall': [],
            # 'train_precision': [], 'val_precision': [],
            # 'train_accuracy': [], 'val_accuracy': []
        }


    def _log_metrics(self, epoch):
        log_dict = {"epoch": epoch}
        for key, values in self.metrics_history.items():
            log_dict[key] = values[-1] if values else None

        df = pd.DataFrame([log_dict])
        if not os.path.exists(self.log_path):
            df.to_csv(self.log_path, index=False)
        else:
            df.to_csv(self.log_path, mode='a', header=False, index=False)

    def evaluate_model(self, df_part):
        self.model.eval()
        dataloader = DataLoader(self.dataset[df_part], batch_size=self.batch_size, shuffle=False, drop_last=False)

        all_predict_proba = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                x_batch, y_batch = batch
                y_batch = y_batch.to(self.device)
                x_num, x_cat = x_batch[0].to(self.device), x_batch[1].to(self.device)

                if self.type_model == "NN":
                    predict = self.model(torch.cat((x_num, x_cat), dim=1))
                elif self.type_model == "MLP-Transformer":
                    predict = self.model(x_cat, x_num)
                else:
                    predict = self.model(x_num, x_cat)

                predict = predict.squeeze(1)
                predict_proba = torch.sigmoid(predict).cpu().numpy()
                target = y_batch.cpu().numpy()

                all_predict_proba.extend(predict_proba)
                all_targets.extend(target)

                loss = self.loss_fn(predict, y_batch)
                total_loss += loss.item()

        roc_auc = roc_auc_score(all_targets, all_predict_proba)
        avg_loss = total_loss / len(dataloader)

        self.metrics_history[f"{df_part}_roc_auc"].append(roc_auc)
        self.metrics_history[f"{df_part}_gini"].append(2 * roc_auc - 1)
        self.metrics_history[f"{df_part}_loss"].append(avg_loss)

        return roc_auc, avg_loss

    def prepare_boosting_data(self, df_part):
        X_num = self.dataset[df_part].data['num']
        X_cat = self.dataset[df_part].data['cat']

        X = np.hstack([X_num, X_cat])
        cat_features = list(range(X_num.shape[1], X.shape[1]))

        y = self.dataset[df_part].labels
        return X, y, cat_features

    def boosting_fit(self):
        X_train, y_train, cat_features = self.prepare_boosting_data('train')
        X_test, y_test, _ = self.prepare_boosting_data('test')

        self.model.fit(
            X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test),
            use_best_model=True
        )

    def boosting_evaluate(self, df_part):
        X_train, y_train, _ = self.prepare_boosting_data(df_part)

        proba = self.model.predict_proba(X_train)[:, 1]
        roc_auc = roc_auc_score(y_train, proba)

        return roc_auc

    def train(self, n_epoch=30, batch_size=128):
        if self.type_model == "Boosting":
            self.boosting_fit()
            train_roc_auc = self.boosting_evaluate('train')
            val_roc_auc = self.boosting_evaluate('val')

            print(f"Train ROC-AUC: {round(train_roc_auc, 4)} \n"
                  f"Train GINI: {round(2 * train_roc_auc - 1, 4)} \n"
                  f"Validate ROC-AUC: {round(val_roc_auc, 4)} \n"
                  f"Validate GINI: {round(2 * val_roc_auc - 1, 4)} \n")

        else:
            self.n_epoch = n_epoch
            self.batch_size = batch_size
            train_dataloader = DataLoader(self.dataset['train'], batch_size=batch_size, shuffle=True, drop_last=True)

            pbar = tqdm(range(1, n_epoch + 1), desc="Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]")
            for epoch in pbar:
                self.model.train()
                epoch_losses = []
                for batch in train_dataloader:
                    self.optimizer.zero_grad()

                    x_batch, y_batch = batch
                    x_num, x_cat = x_batch[0].to(self.device), x_batch[1].to(self.device)
                    y_batch = y_batch.to(self.device)

                    if self.type_model == "NN":
                        predict = self.model(torch.cat((x_num, x_cat), dim=1))
                    elif self.type_model == "MLP-Transformer":
                        predict = self.model(x_cat, x_num)
                    else:
                        predict = self.model(x_num, x_cat)

                    predict = predict.squeeze(1)

                    loss = self.loss_fn(predict, y_batch)
                    loss.backward()

                    epoch_losses.append(loss.detach().cpu().numpy())

                    self.optimizer.step()

                train_roc_auc, train_loss = self.evaluate_model('train')
                test_roc_auc, test_loss = self.evaluate_model('test')

                self._log_metrics(epoch)

                message = (f"{epoch}) Train ROC-AUC: {round(train_roc_auc, 4)} "
                          f"Train Loss: {round(train_loss, 4)}\n"
                          f"   Test ROC-AUC: {round(test_roc_auc, 4)} "
                          f"Test Loss: {round(test_loss, 4)}")

                pbar.clear()
                tqdm.write(message)

                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("ROC-AUC/train", train_roc_auc, epoch)
                self.writer.add_scalar("Loss/val", test_loss, epoch)
                self.writer.add_scalar("ROC-AUC/val", test_roc_auc, epoch)

            self.writer.close()

    def get_metrics_graphic(self):
        epochs_range = range(1, self.n_epoch + 1)
        metrics = ['loss', 'roc_auc', 'gini']

        n_metrics = len(metrics)
        n_graphic_in_col = 2
        n_rows = (n_metrics + n_graphic_in_col - 1) // n_graphic_in_col

        fig, axes = plt.subplots(n_rows, n_graphic_in_col, figsize=(15, 4 * n_rows))

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, metric in enumerate(metrics):
            row = i // n_graphic_in_col
            col = i % n_graphic_in_col

            ax = axes[row, col]
            ax.plot(epochs_range, self.metrics_history[f'train_{metric}'], 'o-', label=f'Train {metric}')
            ax.plot(epochs_range, self.metrics_history[f'test_{metric}'], 'o-', label=f'Test {metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} per epoch')
            ax.legend()

        for i in range(n_metrics, n_rows * n_graphic_in_col):
            row = i // n_graphic_in_col
            col = i % n_graphic_in_col
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()
    
    def compute_shap_values(self, df_part='val', sample_size=1000):
        if self.type_model == "Boosting":
            return self._compute_shap_boosting(df_part, sample_size)
        else:
            return self._compute_shap_nn(df_part, sample_size)
    
    def _compute_shap_boosting(self, df_part, sample_size):
        X, _, _ = self.prepare_boosting_data(df_part)
        
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        return X_sample, shap_values
    
    def _compute_shap_nn(self, df_part, sample_size):
        x_num = self.dataset[df_part].data['num']
        x_cat = self.dataset[df_part].data['cat']

        idx = np.random.choice(len(x_num), min(sample_size, len(x_num)), replace=False)
        x_sample = torch.cat([x_num[idx], x_cat[idx]], dim=1).to(self.device)

        background = x_sample[:50].cpu().numpy()

        def predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return self.model(x_tensor).cpu().numpy()
        
        explainer = shap.KernelExplainer(predict_fn, background)

        shap_values = explainer.shap_values(x_sample.cpu().numpy(), silent=True)

        return x_sample.cpu().numpy(), shap_values

    
    def plot_shap_summary(self, df_part='val', sample_size=1000):
        X, shap_values = self.compute_shap_values(df_part, sample_size)
        
        feature_names = list(self.dataset['train'].data['num'].columns) + \
            list(self.dataset['train'].data['cat'].columns)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot ({self.type_model} model)")
        plt.tight_layout()
        plt.show()