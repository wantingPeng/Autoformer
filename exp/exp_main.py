from utils.logger import logger
from data_provider.data_loader_anomaly import get_anomaly_loader
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, adjust_predictions
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        logger.info(f"Loading {flag} data from {self.args.data_path}")
        try:
            if hasattr(self.args, 'root_path') and self.args.root_path:
                # Use os.path.join to create proper path
                import os
                full_path = os.path.join(self.args.root_path, self.args.data_path)
                logger.info(f"Full data path: {full_path}")
                data_set, data_loader = get_anomaly_loader(
                    data_path=full_path,
                    batch_size=self.args.batch_size,
                    win_size=self.args.seq_len,
                    step=self.args.stride,
                    mode=flag,
                    num_workers=self.args.num_workers
                )
            else:
                data_set, data_loader = get_anomaly_loader(
                    data_path=self.args.data_path,
                    batch_size=self.args.batch_size,
                    win_size=self.args.seq_len,
                    step=self.args.stride,
                    mode=flag,
                    num_workers=self.args.num_workers
                )
            return data_set, data_loader
        except Exception as e:
            logger.error(f"Error loading {flag} data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # For anomaly detection, we still use MSE for reconstruction error
        criterion = nn.MSELoss(reduction='none')
        return criterion

    def _predict(self, batch_x, batch_y):
        # For anomaly detection, we'll use forward pass to predict and compare with input
        # Create dummy input for decoder since we're focused on reconstruction
        batch_size, seq_len, _ = batch_x.shape
        x_mark_enc = torch.zeros((batch_size, seq_len, 4), device=batch_x.device)  # Simple dummy time features
        x_mark_dec = torch.zeros_like(x_mark_enc)
        
        # Use the same input for both encoder and decoder (for reconstruction-based anomaly detection)
        x_dec = batch_x.clone()
        
        def _run_model():
            # Use the standard forward pass but treat output as reconstruction
            outputs = self.model(
                x_enc=batch_x, 
                x_mark_enc=x_mark_enc,
                x_dec=x_dec, 
                x_mark_dec=x_mark_dec
            )
            return outputs
            
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()
            
        return outputs, batch_x

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_predictions = []
        all_true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)  # anomaly labels
                
                outputs, inputs = self._predict(batch_x, batch_y)
                
                # Calculate reconstruction error
                reconstruction_errors = criterion(outputs, inputs)  # batch_size x seq_len x features
                # Mean across feature dimension to get error per timestep
                errors = torch.mean(reconstruction_errors, dim=-1)  # batch_size x seq_len
                
                # Calculate loss - mean across all dimensions
                loss = torch.mean(errors)
                total_loss.append(loss.item())
                
                # Store batch results for anomaly detection metrics
                predictions = (errors > self.args.threshold).float().cpu().numpy()
                true_labels = batch_y.cpu().numpy()
                
                all_predictions.append(predictions)
                all_true_labels.append(true_labels)
                
        # Calculate overall metrics
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()
        all_true_labels = np.concatenate(all_true_labels, axis=0).flatten()
        
        # 计算原始指标
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')
        accuracy = accuracy_score(all_true_labels, all_predictions)
        
        # 应用后处理函数并计算调整后的指标
        adjusted_preds = adjust_predictions(all_predictions, all_true_labels)
        adjusted_precision, adjusted_recall, adjusted_f1, _ = precision_recall_fscore_support(all_true_labels, adjusted_preds, average='binary')
        adjusted_accuracy = accuracy_score(all_true_labels, adjusted_preds)
        
        # Only calculate AUC if we have both positive and negative samples
        if len(np.unique(all_true_labels)) > 1:
            auc = roc_auc_score(all_true_labels, all_predictions)
        else:
            auc = 0.0
        
        total_loss = np.average(total_loss)
        self.model.train()
        
        # 返回原始指标和调整后的指标
        return total_loss, accuracy, precision, recall, f1, auc, adjusted_accuracy, adjusted_precision, adjusted_recall, adjusted_f1

    def train(self, setting):
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"anomaly_detection_{timestamp}.log"
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # Initialize anomaly threshold - this will be adjusted during training
        if not hasattr(self.args, 'threshold'):
            self.args.threshold = 0.1  # Default threshold for anomaly detection

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_errors = []

            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)  # Input sequences
                batch_y = batch_y.float().to(self.device)  # Anomaly labels
                
                # Get model outputs
                outputs, inputs = self._predict(batch_x, batch_y)
                
                # Calculate reconstruction error 
                reconstruction_errors = criterion(outputs, inputs)  # batch_size x seq_len x features
                errors = torch.mean(reconstruction_errors, dim=-1)  # Average over features
                
                # For training, we optimize to minimize reconstruction error
                loss = torch.mean(errors)  # Average over batch and sequence length
                train_loss.append(loss.item())
                
                # Store errors for threshold calculation
                train_errors.extend(errors.detach().cpu().numpy().flatten())

                if (i + 1) % 100 == 0:
                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            
            # Update threshold based on training errors (e.g., 95th percentile)
            if train_errors:
                self.args.threshold = np.percentile(train_errors, 95)
                logger.info(f"Updated anomaly threshold to: {self.args.threshold:.6f}")
            
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # Validate with updated threshold
            vali_results = self.vali(vali_data, vali_loader, criterion)
            test_results = self.vali(test_data, test_loader, criterion)
            
            # 解包结果
            vali_loss, vali_acc, vali_prec, vali_rec, vali_f1, vali_auc, vali_adj_acc, vali_adj_prec, vali_adj_rec, vali_adj_f1 = vali_results
            test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_adj_acc, test_adj_prec, test_adj_rec, test_adj_f1 = test_results

            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps}")
            logger.info(f"Train Loss: {train_loss:.7f}")
            
            # 输出原始指标
            logger.info(f"Valid Loss: {vali_loss:.7f}, Acc: {vali_acc:.4f}, Precision: {vali_prec:.4f}, Recall: {vali_rec:.4f}, F1: {vali_f1:.4f}, AUC: {vali_auc:.4f}")
            logger.info(f"Test Loss: {test_loss:.7f}, Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
            
            # 输出调整后的指标
            logger.info(f"Adjusted Valid - Acc: {vali_adj_acc:.4f}, Precision: {vali_adj_prec:.4f}, Recall: {vali_adj_rec:.4f}, F1: {vali_adj_f1:.4f}")
            logger.info(f"Adjusted Test - Acc: {test_adj_acc:.4f}, Precision: {test_adj_prec:.4f}, Recall: {test_adj_rec:.4f}, F1: {test_adj_f1:.4f}")
            
            # Save model based on validation loss
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            logger.info('Loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        inputs_list = []
        reconstructions_list = []
        errors_list = []
        labels_list = []
        anomaly_preds_list = []
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        criterion = self._select_criterion()
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)  # Input sequences
                batch_y = batch_y.float().to(self.device)  # Anomaly labels
                
                # Get reconstructions
                outputs, inputs = self._predict(batch_x, batch_y)
                
                # Calculate reconstruction error
                reconstruction_errors = criterion(outputs, inputs)  # batch_size x seq_len x features
                errors = torch.mean(reconstruction_errors, dim=-1)  # Average over features
                
                # Determine anomalies based on threshold
                anomaly_preds = (errors > self.args.threshold).float()
                
                # Store batch results
                inputs_list.append(inputs.detach().cpu().numpy())
                reconstructions_list.append(outputs.detach().cpu().numpy())
                errors_list.append(errors.detach().cpu().numpy())
                labels_list.append(batch_y.cpu().numpy())
                anomaly_preds_list.append(anomaly_preds.cpu().numpy())
                
                # Visualize some results
                if i % 20 == 0:
                    # Take first sample in batch
                    sample_idx = 0
                    sample_input = inputs[sample_idx].detach().cpu().numpy()
                    sample_recon = outputs[sample_idx].detach().cpu().numpy()
                    sample_error = errors[sample_idx].detach().cpu().numpy()
                    sample_label = batch_y[sample_idx].cpu().numpy()
                    
                    # Visualize reconstruction and error
                    fig_path = os.path.join(folder_path, f"sample_{i}.pdf")
                    visual(
                        sample_input.mean(axis=1),  # Average across features for visualization
                        sample_recon.mean(axis=1), 
                        fig_path
                    )

        # Concatenate all batches
        all_inputs = np.concatenate(inputs_list, axis=0)
        all_reconstructions = np.concatenate(reconstructions_list, axis=0)
        all_errors = np.concatenate(errors_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        all_preds = np.concatenate(anomaly_preds_list, axis=0)
        
        # Flatten for metrics calculation
        flat_errors = all_errors.flatten()
        flat_labels = all_labels.flatten()
        flat_preds = all_preds.flatten()
        
        # Calculate original metrics
        accuracy = accuracy_score(flat_labels, flat_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(flat_labels, flat_preds, average='binary')
        
        # Only calculate AUC if we have both positive and negative samples
        if len(np.unique(flat_labels)) > 1:
            auc = roc_auc_score(flat_labels, flat_errors)
        else:
            auc = 0.0
        
        # 应用后处理函数并计算调整后的指标
        adjusted_preds = adjust_predictions(flat_preds, flat_labels)
        adjusted_accuracy = accuracy_score(flat_labels, adjusted_preds)
        adjusted_precision, adjusted_recall, adjusted_f1, _ = precision_recall_fscore_support(flat_labels, adjusted_preds, average='binary')
        
        # 输出原始指标
        logger.info("=== Original Test Results ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC: {auc:.4f}")
        
        # 输出调整后的指标
        logger.info("=== Adjusted Test Results ===")
        logger.info(f"Adjusted Accuracy: {adjusted_accuracy:.4f}")
        logger.info(f"Adjusted Precision: {adjusted_precision:.4f}")
        logger.info(f"Adjusted Recall: {adjusted_recall:.4f}")
        logger.info(f"Adjusted F1 Score: {adjusted_f1:.4f}")
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save results
        with open(os.path.join(folder_path, "anomaly_metrics.txt"), 'w') as f:
            f.write(f"Setting: {setting}\n")
            f.write("=== Original Metrics ===\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write("\n=== Adjusted Metrics ===\n")
            f.write(f"Adjusted Accuracy: {adjusted_accuracy:.4f}\n")
            f.write(f"Adjusted Precision: {adjusted_precision:.4f}\n")
            f.write(f"Adjusted Recall: {adjusted_recall:.4f}\n")
            f.write(f"Adjusted F1 Score: {adjusted_f1:.4f}\n")

        # Save arrays
        np.save(folder_path + 'anomaly_metrics.npy', np.array([accuracy, precision, recall, f1, auc]))
        np.save(folder_path + 'adjusted_anomaly_metrics.npy', np.array([adjusted_accuracy, adjusted_precision, adjusted_recall, adjusted_f1]))
        np.save(folder_path + 'adjusted_predictions.npy', adjusted_preds)
        np.save(folder_path + 'inputs.npy', all_inputs)
        np.save(folder_path + 'reconstructions.npy', all_reconstructions)
        np.save(folder_path + 'errors.npy', all_errors)
        np.save(folder_path + 'labels.npy', all_labels)
        np.save(folder_path + 'predictions.npy', all_preds)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='test')  # Use test data for prediction

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logger.info(f"Loading model from {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))

        inputs_list = []
        reconstructions_list = []
        errors_list = []
        anomaly_preds_list = []

        criterion = self._select_criterion()
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)  # Input sequences
                
                # Get reconstructions
                outputs, inputs = self._predict(batch_x, batch_y)
                
                # Calculate reconstruction error
                reconstruction_errors = criterion(outputs, inputs)  # batch_size x seq_len x features
                errors = torch.mean(reconstruction_errors, dim=-1)  # Average over features
                
                # Determine anomalies based on threshold
                anomaly_preds = (errors > self.args.threshold).float()
                
                # Store batch results
                inputs_list.append(inputs.detach().cpu().numpy())
                reconstructions_list.append(outputs.detach().cpu().numpy())
                errors_list.append(errors.detach().cpu().numpy())
                anomaly_preds_list.append(anomaly_preds.cpu().numpy())

        # Concatenate all batches
        all_inputs = np.concatenate(inputs_list, axis=0)
        all_reconstructions = np.concatenate(reconstructions_list, axis=0)
        all_errors = np.concatenate(errors_list, axis=0)
        all_preds = np.concatenate(anomaly_preds_list, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'anomaly_inputs.npy', all_inputs)
        np.save(folder_path + 'anomaly_reconstructions.npy', all_reconstructions)
        np.save(folder_path + 'anomaly_errors.npy', all_errors)
        np.save(folder_path + 'anomaly_predictions.npy', all_preds)

        logger.info(f"Prediction complete. Results saved to {folder_path}")
        return
