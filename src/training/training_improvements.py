import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.0,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.0,
                 last_epoch: int = -1):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            first_cycle_steps: First cycle step size
            cycle_mult: Cycle steps multiplier
            max_lr: First cycle's max learning rate
            min_lr: Min learning rate
            warmup_steps: Linear warmup step size
            gamma: Decrease rate of max learning rate by cycle
            last_epoch: The index of last epoch
        """
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self) -> List[float]:
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.base_lrs]
        
        if self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        
        return [base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            self.step_in_cycle = epoch
            if epoch >= self.cur_cycle_steps:
                self.cycle += 1
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class MixupAugmentation:
    """
    Implementation of mixup augmentation.
    """
    def __init__(self, alpha: float = 0.2):
        """
        Initialize mixup augmentation.
        
        Args:
            alpha: Mixup alpha parameter
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Mixed input, mixed target, and mixing weight
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, lam

class CutMixAugmentation:
    """
    Implementation of CutMix augmentation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix augmentation.
        
        Args:
            alpha: CutMix alpha parameter
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Mixed input, mixed target, and mixing weight
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate random box
        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, lam

class CurriculumLearning:
    """
    Implementation of curriculum learning.
    """
    def __init__(self,
                 num_epochs: int,
                 difficulty_levels: List[float],
                 transition_epochs: List[int]):
        """
        Initialize curriculum learning.
        
        Args:
            num_epochs: Total number of epochs
            difficulty_levels: List of difficulty levels (0-1)
            transition_epochs: List of epochs to transition between levels
        """
        self.num_epochs = num_epochs
        self.difficulty_levels = difficulty_levels
        self.transition_epochs = transition_epochs
        
        # Validate inputs
        assert len(difficulty_levels) == len(transition_epochs) + 1
        assert all(0 <= d <= 1 for d in difficulty_levels)
        assert all(0 <= t < num_epochs for t in transition_epochs)
        assert all(transition_epochs[i] < transition_epochs[i+1] for i in range(len(transition_epochs)-1))
    
    def get_difficulty(self, epoch: int) -> float:
        """
        Get current difficulty level.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Current difficulty level
        """
        for i, transition_epoch in enumerate(self.transition_epochs):
            if epoch < transition_epoch:
                return self.difficulty_levels[i]
        return self.difficulty_levels[-1]
    
    def filter_samples(self,
                      dataset: torch.utils.data.Dataset,
                      epoch: int) -> torch.utils.data.Subset:
        """
        Filter dataset based on current difficulty level.
        
        Args:
            dataset: Original dataset
            epoch: Current epoch
            
        Returns:
            Filtered dataset
        """
        difficulty = self.get_difficulty(epoch)
        
        # Sort samples by difficulty (implement your own difficulty metric)
        difficulties = self._calculate_sample_difficulties(dataset)
        sorted_indices = torch.argsort(difficulties)
        
        # Select samples based on current difficulty level
        num_samples = int(len(dataset) * difficulty)
        selected_indices = sorted_indices[:num_samples]
        
        return torch.utils.data.Subset(dataset, selected_indices)
    
    def _calculate_sample_difficulties(self,
                                     dataset: torch.utils.data.Dataset) -> torch.Tensor:
        """
        Calculate difficulty scores for each sample.
        Implement your own difficulty metric here.
        
        Args:
            dataset: Dataset to calculate difficulties for
            
        Returns:
            Tensor of difficulty scores
        """
        # Example: Use image complexity as difficulty metric
        difficulties = []
        for i in range(len(dataset)):
            img, _ = dataset[i]
            # Calculate image complexity (e.g., variance)
            complexity = torch.var(img).item()
            difficulties.append(complexity)
        
        return torch.tensor(difficulties)

class TrainingImprovements:
    """
    Class implementing various training improvements.
    """
    def __init__(self,
                 model: nn.Module,
                 config: Dict):
        """
        Initialize training improvements.
        
        Args:
            model: Model to train
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize augmentations
        self.mixup = MixupAugmentation(alpha=config.get('mixup_alpha', 0.2))
        self.cutmix = CutMixAugmentation(alpha=config.get('cutmix_alpha', 1.0))
        
        # Initialize curriculum learning
        self.curriculum = CurriculumLearning(
            num_epochs=config['training']['num_epochs'],
            difficulty_levels=config.get('curriculum_levels', [0.3, 0.6, 1.0]),
            transition_epochs=config.get('curriculum_transitions', [10, 20])
        )
    
    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   epoch: int) -> float:
        """
        Train for one epoch with improvements.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['images'].to(self.device)
            labels = batch['labels'].float().to(self.device)
            
            # Apply augmentation
            if np.random.random() < 0.5:
                inputs, labels, lam = self.mixup(inputs, labels)
            else:
                inputs, labels, lam = self.cutmix(inputs, labels)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clipping']['max_norm']
            )
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def get_scheduler(self,
                     optimizer: torch.optim.Optimizer) -> _LRScheduler:
        """
        Get learning rate scheduler with warmup.
        
        Args:
            optimizer: Optimizer
            
        Returns:
            Learning rate scheduler
        """
        return CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=self.config['training']['num_epochs'],
            cycle_mult=1.0,
            max_lr=self.config['training']['learning_rate'],
            min_lr=self.config['training']['scheduler']['min_lr'],
            warmup_steps=int(self.config['training']['num_epochs'] * 0.1),
            gamma=1.0
        )
    
    def get_filtered_dataset(self,
                           dataset: torch.utils.data.Dataset,
                           epoch: int) -> torch.utils.data.Subset:
        """
        Get filtered dataset based on curriculum learning.
        
        Args:
            dataset: Original dataset
            epoch: Current epoch
            
        Returns:
            Filtered dataset
        """
        return self.curriculum.filter_samples(dataset, epoch) 