import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import mlflow
from datetime import datetime

class BiasMonitor:
    def __init__(self, config: Dict):
        """
        Initialize the bias monitoring system.
        
        Args:
            config: Configuration dictionary containing monitoring parameters
        """
        self.config = config
        self.metrics_history = {
            'demographic_metrics': [],
            'feature_importance': [],
            'fairness_metrics': [],
            'performance_metrics': []
        }
        self.demographic_groups = config.get('demographic_groups', [])
        self.sensitive_attributes = config.get('sensitive_attributes', [])
        self.feature_names = config.get('feature_names', [])
        
    def analyze_data_distribution(self, dataset: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of data across demographic and clinical factors.
        
        Args:
            dataset: DataFrame containing demographic and clinical data
            
        Returns:
            Dictionary containing distribution statistics
        """
        distribution_stats = {}
        
        # Analyze demographic distributions
        for group in self.demographic_groups:
            if group in dataset.columns:
                distribution_stats[group] = {
                    'counts': dataset[group].value_counts().to_dict(),
                    'percentages': (dataset[group].value_counts(normalize=True) * 100).to_dict()
                }
        
        # Analyze clinical feature distributions
        for feature in self.feature_names:
            if feature in dataset.columns:
                distribution_stats[feature] = {
                    'mean': dataset[feature].mean(),
                    'std': dataset[feature].std(),
                    'min': dataset[feature].min(),
                    'max': dataset[feature].max(),
                    'missing': dataset[feature].isnull().sum()
                }
        
        # Create visualizations
        self._create_distribution_plots(dataset, distribution_stats)
        
        return distribution_stats
    
    def monitor_model_performance(self, 
                                model: nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                demographic_data: pd.DataFrame) -> Dict:
        """
        Monitor model performance across different demographic groups.
        
        Args:
            model: The trained model
            dataloader: DataLoader containing the evaluation data
            demographic_data: DataFrame containing demographic information
            
        Returns:
            Dictionary containing performance metrics by demographic group
        """
        model.eval()
        performance_metrics = {}
        
        with torch.no_grad():
            for group in self.demographic_groups:
                if group in demographic_data.columns:
                    group_metrics = self._calculate_group_metrics(model, dataloader, demographic_data, group)
                    performance_metrics[group] = group_metrics
        
        # Log metrics to MLflow
        self._log_performance_metrics(performance_metrics)
        
        return performance_metrics
    
    def analyze_feature_importance(self, 
                                 model: nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 feature_names: List[str]) -> Dict:
        """
        Analyze the importance of different features in the model's decisions.
        
        Args:
            model: The trained model
            dataloader: DataLoader containing the evaluation data
            feature_names: List of feature names
            
        Returns:
            Dictionary containing feature importance scores
        """
        feature_importance = {}
        
        # Calculate feature importance using permutation importance
        for feature_idx, feature_name in enumerate(feature_names):
            importance_score = self._calculate_permutation_importance(
                model, dataloader, feature_idx
            )
            feature_importance[feature_name] = importance_score
        
        # Create feature importance visualization
        self._create_feature_importance_plot(feature_importance)
        
        return feature_importance
    
    def calculate_fairness_metrics(self,
                                 predictions: np.ndarray,
                                 true_labels: np.ndarray,
                                 demographic_data: pd.DataFrame) -> Dict:
        """
        Calculate various fairness metrics across demographic groups.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            demographic_data: DataFrame containing demographic information
            
        Returns:
            Dictionary containing fairness metrics
        """
        fairness_metrics = {}
        
        for group in self.demographic_groups:
            if group in demographic_data.columns:
                group_metrics = self._calculate_group_fairness_metrics(
                    predictions, true_labels, demographic_data, group
                )
                fairness_metrics[group] = group_metrics
        
        # Log fairness metrics to MLflow
        self._log_fairness_metrics(fairness_metrics)
        
        return fairness_metrics
    
    def create_monitoring_dashboard(self,
                                  performance_metrics: Dict,
                                  fairness_metrics: Dict,
                                  feature_importance: Dict) -> None:
        """
        Create a dashboard for monitoring bias metrics over time.
        
        Args:
            performance_metrics: Dictionary containing performance metrics
            fairness_metrics: Dictionary containing fairness metrics
            feature_importance: Dictionary containing feature importance scores
        """
        # Create timestamp for the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the dashboard figure
        fig = plt.figure(figsize=(20, 15))
        
        # Plot performance metrics
        self._plot_performance_metrics(performance_metrics, fig, 1)
        
        # Plot fairness metrics
        self._plot_fairness_metrics(fairness_metrics, fig, 2)
        
        # Plot feature importance
        self._plot_feature_importance(feature_importance, fig, 3)
        
        # Save the dashboard
        plt.tight_layout()
        plt.savefig(f'bias_monitoring_dashboard_{timestamp}.png')
        plt.close()
        
        # Log the dashboard to MLflow
        mlflow.log_artifact(f'bias_monitoring_dashboard_{timestamp}.png')
    
    def _calculate_group_metrics(self,
                               model: nn.Module,
                               dataloader: torch.utils.data.DataLoader,
                               demographic_data: pd.DataFrame,
                               group: str) -> Dict:
        """Calculate performance metrics for a specific demographic group."""
        group_metrics = {}
        
        for value in demographic_data[group].unique():
            # Get indices for this group
            group_indices = demographic_data[demographic_data[group] == value].index
            
            # Calculate metrics for this group
            group_predictions = []
            group_labels = []
            
            for batch in dataloader:
                if batch['indices'][0] in group_indices:
                    outputs = model(batch['images'], batch['positions'], batch['attention_mask'])
                    predictions = torch.sigmoid(outputs).cpu().numpy()
                    group_predictions.extend(predictions)
                    group_labels.extend(batch['labels'].cpu().numpy())
            
            if len(group_predictions) > 0:
                group_metrics[value] = {
                    'accuracy': accuracy_score(group_labels, np.round(group_predictions)),
                    'precision': precision_score(group_labels, np.round(group_predictions)),
                    'recall': recall_score(group_labels, np.round(group_predictions)),
                    'f1': f1_score(group_labels, np.round(group_predictions)),
                    'auc': roc_auc_score(group_labels, group_predictions)
                }
        
        return group_metrics
    
    def _calculate_permutation_importance(self,
                                        model: nn.Module,
                                        dataloader: torch.utils.data.DataLoader,
                                        feature_idx: int) -> float:
        """Calculate permutation importance for a specific feature."""
        # Store original predictions
        original_predictions = []
        original_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(batch['images'], batch['positions'], batch['attention_mask'])
                predictions = torch.sigmoid(outputs).cpu().numpy()
                original_predictions.extend(predictions)
                original_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate baseline performance
        baseline_score = roc_auc_score(original_labels, original_predictions)
        
        # Permute the feature and calculate new performance
        permuted_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Permute the feature
                permuted_batch = batch.copy()
                permuted_batch['images'][:, :, :, feature_idx] = torch.randn_like(
                    permuted_batch['images'][:, :, :, feature_idx]
                )
                
                outputs = model(
                    permuted_batch['images'],
                    permuted_batch['positions'],
                    permuted_batch['attention_mask']
                )
                predictions = torch.sigmoid(outputs).cpu().numpy()
                permuted_predictions.extend(predictions)
        
        # Calculate importance as the difference in performance
        permuted_score = roc_auc_score(original_labels, permuted_predictions)
        importance = baseline_score - permuted_score
        
        return importance
    
    def _calculate_group_fairness_metrics(self,
                                        predictions: np.ndarray,
                                        true_labels: np.ndarray,
                                        demographic_data: pd.DataFrame,
                                        group: str) -> Dict:
        """Calculate fairness metrics for a specific demographic group."""
        group_metrics = {}
        
        for value in demographic_data[group].unique():
            # Get indices for this group
            group_indices = demographic_data[demographic_data[group] == value].index
            
            # Calculate metrics for this group
            group_predictions = predictions[group_indices]
            group_labels = true_labels[group_indices]
            
            if len(group_predictions) > 0:
                group_metrics[value] = {
                    'demographic_parity': self._calculate_demographic_parity(group_predictions),
                    'equal_opportunity': self._calculate_equal_opportunity(group_predictions, group_labels),
                    'predictive_parity': self._calculate_predictive_parity(group_predictions, group_labels)
                }
        
        return group_metrics
    
    def _calculate_demographic_parity(self, predictions: np.ndarray) -> float:
        """Calculate demographic parity (statistical parity)."""
        return np.mean(predictions > 0.5)
    
    def _calculate_equal_opportunity(self,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray) -> float:
        """Calculate equal opportunity (true positive rate)."""
        return recall_score(true_labels, predictions > 0.5)
    
    def _calculate_predictive_parity(self,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray) -> float:
        """Calculate predictive parity (positive predictive value)."""
        return precision_score(true_labels, predictions > 0.5)
    
    def _create_distribution_plots(self,
                                 dataset: pd.DataFrame,
                                 distribution_stats: Dict) -> None:
        """Create plots for data distribution analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create demographic distribution plots
        for group in self.demographic_groups:
            if group in dataset.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=dataset, x=group)
                plt.title(f'Distribution of {group}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'demographic_distribution_{group}_{timestamp}.png')
                plt.close()
                
                mlflow.log_artifact(f'demographic_distribution_{group}_{timestamp}.png')
        
        # Create clinical feature distribution plots
        for feature in self.feature_names:
            if feature in dataset.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=dataset, x=feature)
                plt.title(f'Distribution of {feature}')
                plt.tight_layout()
                plt.savefig(f'feature_distribution_{feature}_{timestamp}.png')
                plt.close()
                
                mlflow.log_artifact(f'feature_distribution_{feature}_{timestamp}.png')
    
    def _create_feature_importance_plot(self, feature_importance: Dict) -> None:
        """Create a plot for feature importance analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(12, 6))
        features = list(feature_importance.keys())
        importance_scores = list(feature_importance.values())
        
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)
        features = [features[i] for i in sorted_idx]
        importance_scores = [importance_scores[i] for i in sorted_idx]
        
        plt.barh(range(len(features)), importance_scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{timestamp}.png')
        plt.close()
        
        mlflow.log_artifact(f'feature_importance_{timestamp}.png')
    
    def _plot_performance_metrics(self,
                                performance_metrics: Dict,
                                fig: plt.Figure,
                                subplot_idx: int) -> None:
        """Plot performance metrics in the dashboard."""
        ax = fig.add_subplot(3, 1, subplot_idx)
        
        for group, metrics in performance_metrics.items():
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    for subgroup, score in values.items():
                        ax.bar(f'{group}_{subgroup}_{metric_name}', score)
        
        ax.set_title('Performance Metrics by Demographic Group')
        ax.set_ylabel('Score')
        plt.xticks(rotation=45)
    
    def _plot_fairness_metrics(self,
                             fairness_metrics: Dict,
                             fig: plt.Figure,
                             subplot_idx: int) -> None:
        """Plot fairness metrics in the dashboard."""
        ax = fig.add_subplot(3, 1, subplot_idx)
        
        for group, metrics in fairness_metrics.items():
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    for subgroup, score in values.items():
                        ax.bar(f'{group}_{subgroup}_{metric_name}', score)
        
        ax.set_title('Fairness Metrics by Demographic Group')
        ax.set_ylabel('Score')
        plt.xticks(rotation=45)
    
    def _plot_feature_importance(self,
                               feature_importance: Dict,
                               fig: plt.Figure,
                               subplot_idx: int) -> None:
        """Plot feature importance in the dashboard."""
        ax = fig.add_subplot(3, 1, subplot_idx)
        
        features = list(feature_importance.keys())
        importance_scores = list(feature_importance.values())
        
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)
        features = [features[i] for i in sorted_idx]
        importance_scores = [importance_scores[i] for i in sorted_idx]
        
        ax.barh(range(len(features)), importance_scores)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
    
    def _log_performance_metrics(self, performance_metrics: Dict) -> None:
        """Log performance metrics to MLflow."""
        for group, metrics in performance_metrics.items():
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    for subgroup, score in values.items():
                        mlflow.log_metric(
                            f'performance_{group}_{subgroup}_{metric_name}',
                            score
                        )
    
    def _log_fairness_metrics(self, fairness_metrics: Dict) -> None:
        """Log fairness metrics to MLflow."""
        for group, metrics in fairness_metrics.items():
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    for subgroup, score in values.items():
                        mlflow.log_metric(
                            f'fairness_{group}_{subgroup}_{metric_name}',
                            score
                        ) 