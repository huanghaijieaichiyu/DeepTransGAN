import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn as nn
import time
from rich.console import Console
from rich.table import Table
from rich.progress import (Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn)

from models.cut_networks import (CUTGenerator, FastCUTGenerator, PatchSampleF, PatchNCELoss)
from models.base_mode import Discriminator


class CUTTrainer:
    """
    CUT (Contrastive Unpaired Translation) Trainer.
    
    This class implements the training method described in:
    "Contrastive Learning for Unpaired Image-to-Image Translation"
    Paper: https://arxiv.org/abs/2007.15651
    
    Uses contrastive learning with PatchNCE loss for unpaired domain mapping.
    """

    def __init__(self, args, generator, discriminator=None, fast_mode=False):
        """
        Initialize the CUT trainer.
        
        Args:
            args: Arguments containing training parameters
            generator: The generator model
            discriminator: The discriminator model
            fast_mode: Whether to use FastCUT mode (fewer NCE losses)
        """
        self.args = args
        self.generator = generator
        self.discriminator = discriminator

        # Set device
        self.device = torch.device('cpu')
        if args.device == 'cuda':
            cuda_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(cuda_device)

        # Move models to device
        self.generator = self.generator.to(self.device)
        if self.discriminator:
            self.discriminator = self.discriminator.to(self.device)

        # CUT-specific settings
        self.fast_mode = fast_mode
        self.nce_layers = [0, 4, 8, 12, 16] if not fast_mode else [0, 4, 8]
        self.nce_includes_all_negatives_from_minibatch = False

        # Set temperature for NCE loss
        default_temp = 0.07
        self.nce_temp = args.nce_temp if hasattr(args, 'nce_temp') else default_temp

        # Initialize patch sampler with dummy features
        self.patch_sampler = PatchSampleF(use_mlp=True, nc=256)
        self.patch_sampler = self.patch_sampler.to(self.device)

        # NCE loss
        self.criterion_nce = PatchNCELoss(args.batch_size, self.nce_temp)
        self.criterion_nce = self.criterion_nce.to(self.device)

        # Loss weights
        self.lambda_gan = 1.0
        self.lambda_nce = args.nce_weight if hasattr(args, 'nce_weight') else (1.0 if not fast_mode else 10.0)
        self.lambda_identity = args.identity_weight if hasattr(args,
                                                               'identity_weight') else (0.1 if not fast_mode else 0.0)

        # Initialize optimizers for generator and discriminator
        # Generator optimizer
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        # Discriminator optimizer (if discriminator exists)
        self.d_optimizer = None
        if self.discriminator:
            self.d_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=args.lr * 0.5,  # Lower LR for discriminator
                betas=(args.beta1, args.beta2))

        # Setting up data loaders - these would be passed from outside
        self.train_loader = None
        self.test_loader = None

        # Training parameters
        self.epochs = args.epochs
        self.epoch = 0
        self.grad_clip_generator = 5.0
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.patience = args.patience
        self.patience_counter = 0

        # Paths and logging
        self.path = args.save_path
        os.makedirs(self.path, exist_ok=True)
        self.train_log = os.path.join(self.path, 'train_log.txt')

        # Rich console support
        try:
            self.rich_available = True
            self.console = Console()
        except ImportError:
            self.rich_available = False
            print("Rich library not available. Using standard progress bar.")

        # Print initialization info
        self.log_message(f"Initialized {'FastCUT' if fast_mode else 'CUT'} Trainer")
        self.log_message(f"NCE Layers: {self.nce_layers}")
        self.log_message(f"Lambda GAN: {self.lambda_gan}")
        self.log_message(f"Lambda NCE: {self.lambda_nce}")
        self.log_message(f"Lambda Identity: {self.lambda_identity}")

        # Initialize patch sampler as the final step
        self.patch_sampler_optimizer = None  # Initialize to None first
        self.init_patch_sampler()

    def log_message(self, message):
        """Log a message to console and file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        if hasattr(self, 'rich_available') and self.rich_available:
            self.console.print(f"[cyan]{log_message}[/cyan]")
        else:
            print(log_message)

        with open(self.train_log, 'a') as f:
            f.write(log_message + '\n')

    def create_rich_progress(self):
        """Create a rich progress bar if available"""
        if not hasattr(self, 'rich_available') or not self.rich_available:
            return None

        return Progress(TextColumn("[bold blue]{task.description}"), BarColumn(bar_width=40), TaskProgressColumn(),
                        TimeElapsedColumn(), TimeRemainingColumn())

    def print_epoch_summary(self, epoch, gen_loss, dis_loss, metrics=None):
        """Print a summary of the epoch results"""
        avg_gen_loss = np.mean(gen_loss)
        avg_dis_loss = np.mean(dis_loss) if dis_loss else 0.0

        if hasattr(self, 'rich_available') and self.rich_available:
            # Create table
            table = Table(title=f"📊 Training Summary - Epoch {epoch + 1}/{self.epochs}", expand=True)

            # Add columns
            table.add_column("Category", style="cyan")
            table.add_column("Metric", style="magenta")
            table.add_column("Value", style="green")

            # Add rows
            table.add_row("Loss", "Generator", f"{avg_gen_loss:.6f}")
            if dis_loss:
                table.add_row("Loss", "Discriminator", f"{avg_dis_loss:.6f}")

            # Add learning rates
            g_lr = self.g_optimizer.param_groups[0]['lr']
            table.add_row("Learning Rate", "Generator", f"{g_lr:.6f}")
            if self.d_optimizer:
                d_lr = self.d_optimizer.param_groups[0]['lr']
                table.add_row("Learning Rate", "Discriminator", f"{d_lr:.6f}")

            # Add metrics if provided
            if metrics:
                for key, value in metrics.items():
                    table.add_row("Metrics", key, f"{value:.4f}" if isinstance(value, float) else str(value))

            # Add GPU memory info
            if torch.cuda.is_available():
                table.add_row("GPU Memory", "Allocated", f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                table.add_row("GPU Memory", "Cached", f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            # Print table
            self.console.print()
            self.console.print(table)
            self.console.print()
        else:
            separator = "=" * 80
            print(f"\n{separator}")
            print(f"Epoch {epoch + 1}/{self.epochs} Summary:")
            print(f"Generator Loss: {avg_gen_loss:.6f}")
            if dis_loss:
                print(f"Discriminator Loss: {avg_dis_loss:.6f}")

            # Learning rates
            print(f"\nLearning Rates:")
            print(f"Generator: {self.g_optimizer.param_groups[0]['lr']:.6f}")
            if self.d_optimizer:
                print(f"Discriminator: {self.d_optimizer.param_groups[0]['lr']:.6f}")

            # Metrics
            if metrics:
                print(f"\nMetrics:")
                for key, value in metrics.items():
                    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

            # GPU Memory
            if torch.cuda.is_available():
                print(f"\nGPU Memory Usage:")
                print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            print(f"{separator}\n")

    def _get_features(self, x, get_intermediate=False):
        """
        Extract features from the generator at different layers.
        
        Args:
            x: Input image
            get_intermediate: Whether to return intermediate features
            
        Returns:
            A list of features at different layers
        """
        features = []

        # Initial feature extraction
        feat = self.generator.conv1(x)
        if 0 in self.nce_layers:
            features.append(feat)

        # Encoder features
        feat = self.generator.down1(feat)
        if 4 in self.nce_layers:
            features.append(feat)

        feat = self.generator.down2(feat)
        if 8 in self.nce_layers:
            features.append(feat)

        feat = self.generator.down3(feat)
        if 12 in self.nce_layers:
            features.append(feat)

        feat = self.generator.bottleneck(feat)
        if 16 in self.nce_layers:
            features.append(feat)

        # Return intermediate or final features
        if get_intermediate:
            return features
        return features[-1]  # Return only the last feature

    def init_patch_sampler(self):
        """Initialize the patch sampler optimizer"""
        try:
            # PatchSampleF is now initialized with default layers in __init__
            # So we can directly create the optimizer
            self.patch_sampler_optimizer = torch.optim.Adam(
                self.patch_sampler.parameters(),
                lr=self.args.lr * 0.1,  # Lower learning rate for the patch sampler
                betas=(self.args.beta1, self.args.beta2))
            self.log_message("Patch sampler optimizer initialized successfully")
        except Exception as e:
            self.log_message(f"Error initializing patch sampler optimizer: {e}")
            # Create a minimal optimizer with any available parameters
            params = [p for p in self.patch_sampler.parameters() if p.requires_grad]
            if len(params) > 0:
                self.patch_sampler_optimizer = torch.optim.Adam(params, lr=self.args.lr * 0.1)
                self.log_message(f"Created fallback optimizer with {len(params)} parameters")
            else:
                self.log_message("No parameters found in patch sampler, creating dummy optimizer")
                # Create dummy parameter if none exists
                if not hasattr(self.patch_sampler, 'dummy_param'):
                    self.patch_sampler.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))
                self.patch_sampler_optimizer = torch.optim.Adam([self.patch_sampler.dummy_param], lr=self.args.lr * 0.1)

    def compute_nce_loss(self, real, fake):
        """
        Compute the PatchNCE loss.
        
        Args:
            real: Real image
            fake: Generated image
            
        Returns:
            NCE loss
        """
        # Get features from the generator
        real_features = self._get_features(real, get_intermediate=True)
        fake_features = self._get_features(fake, get_intermediate=True)

        # Sample patches and compute NCE loss
        feat_k_pool, sample_ids = self.patch_sampler(real_features, num_patches=256)
        feat_q_pool, _ = self.patch_sampler(fake_features, num_patches=256, patch_ids=sample_ids)

        total_nce_loss = 0.0
        n_layers = len(feat_k_pool)

        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            nce_loss = self.criterion_nce(f_q, f_k)
            total_nce_loss += nce_loss

        return total_nce_loss / n_layers

    def compute_generator_loss(self, fake_outputs, fake_images, real_images):
        """
        Compute the generator loss for CUT/FastCUT.
        
        Args:
            fake_outputs: Discriminator outputs for fake images
            fake_images: Generated images
            real_images: Real images
            
        Returns:
            Total generator loss, individual loss components
        """
        # GAN loss
        gan_loss = -fake_outputs.mean()

        # NCE loss
        nce_loss = self.compute_nce_loss(real_images, fake_images)

        # Identity preservation loss (optional)
        identity_loss = 0.0
        if self.lambda_identity > 0:
            identity_images = self.generator(real_images)
            identity_loss = F.l1_loss(identity_images, real_images)

        # Total loss
        total_loss = (self.lambda_gan * gan_loss + self.lambda_nce * nce_loss + self.lambda_identity * identity_loss)

        return total_loss, {
            'gan': gan_loss.item(),
            'nce': nce_loss.item(),
            'identity': identity_loss if isinstance(identity_loss, float) else identity_loss.item()
        }

    def add_noise_to_input(self, tensor, noise_factor=None):
        """Add random noise to input tensor for improved stability"""
        if noise_factor is None:
            noise_factor = 0.05
        return tensor + torch.randn_like(tensor) * noise_factor

    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        if self.discriminator:
            self.discriminator.train()
        self.patch_sampler.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        gen_loss = []
        dis_loss = []
        last_batch = None

        scaler_g = GradScaler('cuda')
        scaler_d = GradScaler('cuda')

        # Create progress bar
        if hasattr(self, 'rich_available') and self.rich_available:
            progress = self.create_rich_progress()
            if progress:
                task_id = progress.add_task(f"[cyan]Epoch {self.epoch + 1}/{self.epochs}", total=len(self.train_loader))
                progress.start()
        else:
            pbar = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader),
                        desc=f"Epoch {self.epoch + 1}/{self.epochs}",
                        bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        colour='green')

        for batch_idx, (low_images, high_images) in enumerate(self.train_loader):
            # Move data to device
            low_images = low_images.to(self.device)
            high_images = high_images.to(self.device)

            # Train discriminator
            if self.discriminator:
                self.d_optimizer.zero_grad()

                with autocast('cuda'):
                    # Generate fake images
                    fake_images = self.generator(low_images)

                    # Get discriminator outputs
                    real_outputs = self.discriminator(high_images)
                    fake_outputs = self.discriminator(fake_images.detach())

                    # Compute discriminator loss
                    d_real = F.binary_cross_entropy_with_logits(real_outputs, torch.ones_like(real_outputs))
                    d_fake = F.binary_cross_entropy_with_logits(fake_outputs, torch.zeros_like(fake_outputs))
                    d_loss = (d_real + d_fake) / 2

                scaler_d.scale(d_loss).backward()
                scaler_d.step(self.d_optimizer)
                scaler_d.update()

                running_d_loss += d_loss.item()
                dis_loss.append(d_loss.item())

            # Train generator and patch sampler
            self.g_optimizer.zero_grad()
            if self.patch_sampler_optimizer:
                self.patch_sampler_optimizer.zero_grad()

            with autocast('cuda'):
                # Generate fake images
                fake_images = self.generator(low_images)

                # Compute generator loss
                if self.discriminator:
                    fake_outputs = self.discriminator(fake_images)
                    g_loss, loss_components = self.compute_generator_loss(fake_outputs, fake_images, high_images)
                else:
                    g_loss = self.compute_nce_loss(high_images, fake_images)

            scaler_g.scale(g_loss).backward()
            scaler_g.step(self.g_optimizer)

            # Handle patch sampler optimizer
            if self.patch_sampler_optimizer:
                try:
                    has_grads = any(p.grad is not None for p in self.patch_sampler.parameters() if p.requires_grad)
                    if has_grads:
                        scaler_g.step(self.patch_sampler_optimizer)
                except AssertionError as e:
                    self.patch_sampler_optimizer.step()
                    self.log_message(f"Info: Used regular optimizer step for patch sampler: {e}")

            scaler_g.update()

            running_g_loss += g_loss.item()
            gen_loss.append(g_loss.item())

            # Update progress information
            if batch_idx % 10 == 0:
                lr_g = self.g_optimizer.param_groups[0]['lr']
                lr_d = (self.d_optimizer.param_groups[0]['lr'] if self.d_optimizer else 0.0)

                progress_info = (f"G: {g_loss.item():.4f}, "
                                 f"D: {d_loss.item():.4f} if self.discriminator else '', "
                                 f"LR_G: {lr_g:.6f}, "
                                 f"LR_D: {lr_d:.6f}" if self.d_optimizer else '')

                if (hasattr(self, 'rich_available') and self.rich_available and progress and task_id is not None):
                    progress.update(task_id,
                                    advance=1,
                                    description=(f"[cyan]Epoch {self.epoch + 1}/{self.epochs} - "
                                                 f"{progress_info}"))
                elif 'pbar' in locals():
                    pbar.set_description(f"🔄 {progress_info}")

            last_batch = (low_images, high_images, fake_images)

        # Close progress bar
        if hasattr(self, 'rich_available') and self.rich_available and progress:
            progress.stop()
        elif 'pbar' in locals():
            pbar.close()

        # Calculate average losses
        avg_g_loss = running_g_loss / len(self.train_loader)
        avg_d_loss = (running_d_loss / len(self.train_loader) if self.discriminator else 0.0)

        return avg_g_loss, avg_d_loss, last_batch

    def train(self, train_loader, test_loader=None, num_epochs=None):
        """
        Train the model for a specific number of epochs
        """
        self.train_loader = train_loader
        self.test_loader = test_loader

        if num_epochs is not None:
            self.epochs = num_epochs

        print(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epoch, self.epochs):
            gen_loss, dis_loss, last_batch = self.train_epoch()

            self.print_epoch_summary(epoch, gen_loss, dis_loss)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()

        # Save final model
        self.save_checkpoint()
        print("Training completed!")

    def save_checkpoint(self, is_best=False):
        """
        Save the model checkpoints including the patch sampler.
        """
        # Create directories if they don't exist
        generator_dir = os.path.join(self.path, 'generator')
        patch_sampler_dir = os.path.join(self.path, 'patch_sampler')

        os.makedirs(generator_dir, exist_ok=True)
        os.makedirs(patch_sampler_dir, exist_ok=True)

        if self.discriminator:
            discriminator_dir = os.path.join(self.path, 'discriminator')
            os.makedirs(discriminator_dir, exist_ok=True)

        # Save generator
        g_save_path = os.path.join(generator_dir, 'best.pt' if is_best else 'last.pt')

        g_checkpoint = {
            'net': self.generator.state_dict(),
            'optimizer': self.g_optimizer.state_dict(),
            'epoch': self.epoch
        }

        torch.save(g_checkpoint, g_save_path)

        # Save discriminator if it exists
        if self.discriminator:
            d_save_path = os.path.join(discriminator_dir, 'best.pt' if is_best else 'last.pt')

            d_checkpoint = {
                'net': self.discriminator.state_dict(),
                'optimizer': self.d_optimizer.state_dict() if self.d_optimizer else None,
                'epoch': self.epoch
            }

            torch.save(d_checkpoint, d_save_path)

        # Save patch sampler
        ps_save_path = os.path.join(patch_sampler_dir, 'best.pt' if is_best else 'last.pt')

        ps_checkpoint = {
            'net': self.patch_sampler.state_dict(),
            'optimizer': self.patch_sampler_optimizer.state_dict(),
            'epoch': self.epoch
        }

        torch.save(ps_checkpoint, ps_save_path)

        print(f"Checkpoint saved at epoch {self.epoch}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoints including the patch sampler.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path {checkpoint_path} doesn't exist")
            return

        # Load generator checkpoint
        g_path_checkpoint = os.path.join(checkpoint_path, 'generator/last.pt')
        if os.path.exists(g_path_checkpoint):
            try:
                g_checkpoint = torch.load(g_path_checkpoint, map_location=self.device, weights_only=True)
            except:
                g_checkpoint = torch.load(g_path_checkpoint, map_location=self.device, weights_only=False)

            self.generator.load_state_dict(g_checkpoint['net'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            self.epoch = g_checkpoint['epoch']

            print(f"Loaded generator checkpoint from epoch {self.epoch}")

        # Load discriminator checkpoint
        if self.discriminator:
            d_path_checkpoint = os.path.join(checkpoint_path, 'discriminator/last.pt')
            if os.path.exists(d_path_checkpoint):
                try:
                    d_checkpoint = torch.load(d_path_checkpoint, map_location=self.device, weights_only=True)
                except:
                    d_checkpoint = torch.load(d_path_checkpoint, map_location=self.device, weights_only=False)

                self.discriminator.load_state_dict(d_checkpoint['net'])
                if self.d_optimizer:
                    self.d_optimizer.load_state_dict(d_checkpoint['optimizer'])

                print("Loaded discriminator checkpoint")

        # Load patch sampler checkpoint
        ps_path_checkpoint = os.path.join(checkpoint_path, 'patch_sampler/last.pt')
        if os.path.exists(ps_path_checkpoint):
            try:
                ps_checkpoint = torch.load(ps_path_checkpoint, map_location=self.device, weights_only=True)
            except:
                ps_checkpoint = torch.load(ps_path_checkpoint, map_location=self.device, weights_only=False)

            self.patch_sampler.load_state_dict(ps_checkpoint['net'])
            self.patch_sampler_optimizer.load_state_dict(ps_checkpoint['optimizer'])

            print("Loaded patch sampler checkpoint")


def train_cut(args, fast_mode=False):
    """
    Train a CUT or FastCUT model.
    
    Args:
        args: Training arguments
        fast_mode: Whether to use FastCUT mode
    """
    # Import these here to avoid circular imports
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets.data_set import LowLightDataset

    # Choose generator type
    if fast_mode:
        generator = FastCUTGenerator()
    else:
        generator = CUTGenerator()

    # Create discriminator
    discriminator = Discriminator()

    # Create trainer
    trainer = CUTTrainer(args, generator, discriminator, fast_mode=fast_mode)

    # Create data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_data = LowLightDataset(image_dir=args.data, transform=transform, phase="train")
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    test_data = LowLightDataset(image_dir=args.data, transform=test_transform, phase="test")
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    # Resume training if needed
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train(train_loader, test_loader)
