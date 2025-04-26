import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import pkg_resources
from models.cut_networks import (CUTGenerator, FastCUTGenerator, PatchSampleF, PatchNCELoss)
from models.base_mode import Discriminator
from models.Repvit import RepViTBlock
from models.common import SPPELAN, PSA, Gencov
from utils.misic import set_random_seed
from datasets.data_set import LowLightDataset

# Import BaseTrainer from DeepTranserGAN
from DeepTranserGAN import BaseTrainer


def spectral_norm(module, name='weight', power_iterations=1):
    """Apply spectral normalization to a module's weight."""
    try:
        weight = getattr(module, name)
    except AttributeError:
        # Break the f-string for line length
        msg = (f"Module {module} does not have an "
               f"attribute named {name}")
        raise ValueError(msg)

    if not isinstance(weight, nn.Parameter):
        return module

    original_forward = module.forward

    def _l2normalize(v, eps=1e-12):
        return v / (torch.norm(v) + eps)

    height = weight.data.shape[0]
    # Register u as a buffer attached to the module
    module.register_buffer('sn_u', torch.randn(height, device=weight.device))

    def spectral_norm_forward(*args, **kwargs):
        u = module.sn_u
        weight_mat = weight.view(height, -1)

        with torch.no_grad():
            for _ in range(power_iterations):
                # Power method: Estimate the spectral norm
                v = _l2normalize(torch.matmul(weight_mat.t(), u))
                u = _l2normalize(torch.matmul(weight_mat, v))

            sigma = torch.dot(u, torch.matmul(weight_mat, v))

        # Store original weight data
        original_weight_data = weight.data.clone()

        # Normalize weight by modifying the data directly
        weight.data = weight.data / sigma

        # Call the original forward with normalized weights
        result = original_forward(*args, **kwargs)

        # Restore the original weight data after the forward pass
        weight.data = original_weight_data

        # Update the spectral norm estimate for the next iteration
        module.sn_u.copy_(u.detach())

        return result

    module.forward = spectral_norm_forward
    return module


class SpectralNormConv2d(nn.Module):
    """2D convolution with spectral normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # Break line for length
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


class WarmupCosineScheduler:
    """Warmup cosine learning rate scheduler."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, warmup_start_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            if self.warmup_epochs > 0:
                lr_ratio = self.current_epoch / self.warmup_epochs
            else:
                lr_ratio = 1.0
            lr = (self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * lr_ratio)
        else:
            # Cosine annealing
            denom = (self.total_epochs - self.warmup_epochs)
            if denom > 0:
                # Break line 95
                progress_ratio = ((self.current_epoch - self.warmup_epochs) / denom)
            else:
                progress_ratio = 1.0
            # Break line 98
            cosine_term = 0.5 * (1 + math.cos(math.pi * progress_ratio))
            lr = (self.min_lr + (self.base_lr - self.min_lr) * cosine_term)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'min_lr': self.min_lr,
            'warmup_start_lr': self.warmup_start_lr
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.min_lr = state_dict['min_lr']
        self.warmup_start_lr = state_dict['warmup_start_lr']


class CUTTrainer(BaseTrainer):
    """
    CUT (Contrastive Unpaired Translation) Trainer.
    Extends BaseTrainer to implement the training method described in:
    "Contrastive Learning for Unpaired Image-to-Image Translation"
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
        # Initialize the BaseTrainer with the given arguments and models
        # Assuming BaseTrainer handles optimizer creation based on args.optimizer
        super().__init__(args, generator, discriminator)

        # Apply spectral normalization to discriminator if exists
        if self.discriminator:
            self._apply_spectral_norm(self.discriminator)

        # Initialize warmup scheduler
        warmup_epochs = int(0.1 * args.epochs)  # 10% warmup
        # Break lines for length
        self.scheduler_g = WarmupCosineScheduler(
            optimizer=self.g_optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=args.epochs,
            min_lr=args.lr * 0.001,
            warmup_start_lr=args.lr * 0.0001  # Break line
        )

        self.scheduler_d = None
        if self.d_optimizer:
            # Break lines for length
            self.scheduler_d = WarmupCosineScheduler(
                optimizer=self.d_optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=args.epochs,
                min_lr=args.lr * 0.001,
                warmup_start_lr=args.lr * 0.0001  # Break line
            )

        # CUT-specific settings
        self.fast_mode = fast_mode
        # Break line
        self.nce_layers = [0, 4, 8, 12, 16] if not fast_mode else [0, 4, 8]
        self.nce_includes_all_negatives_from_minibatch = False

        # Set temperature for NCE loss
        default_temp = 0.07
        self.nce_temp = getattr(args, 'nce_temp', default_temp)

        # Initialize patch sampler - Break line
        self.patch_sampler = PatchSampleF(use_mlp=True, nc=256)
        self.patch_sampler = self.patch_sampler.to(self.device)

        # Initialize NCE loss
        # Break line
        self.criterion_nce = PatchNCELoss(args.batch_size, self.nce_temp)
        self.criterion_nce = self.criterion_nce.to(self.device)

        # Loss weights
        self.lambda_gan = getattr(args, 'gan_weight', 1.0)
        default_nce_weight = 1.0 if not fast_mode else 10.0
        default_id_weight = 0.1 if not fast_mode else 0.0
        # Re-break line 179 (was ~187)
        self.lambda_nce = getattr(args, 'nce_weight', default_nce_weight)
        # Re-break line 124 (was ~188)
        self.lambda_identity = getattr(args, 'identity_weight', default_id_weight)

        # Initialize optimizer for the patch sampler
        self.patch_sampler_optimizer = None
        self.init_patch_sampler()  # Call init after defining members

        # Add gradient clipping parameters
        self.grad_clip_generator = getattr(args, 'grad_clip_gen', 5.0)
        # Break line 128 (previous 194)
        self.grad_clip_discriminator = getattr(args, 'grad_clip_disc', 1.0)

        # Add noise parameters for training stability
        self.noise_factor = getattr(args, 'noise_factor', 0.05)

        # Initialize weights
        self._initialize_weights(self.generator)
        if self.discriminator:
            self._initialize_weights(self.discriminator)
        self._initialize_weights(self.patch_sampler)

        # Log initialization info
        mode_str = 'FastCUT' if fast_mode else 'CUT'
        self.log_message(f"Initialized {mode_str} Trainer")
        self.log_message(f"NCE Layers: {self.nce_layers}")
        self.log_message(f"Lambda GAN: {self.lambda_gan}")
        self.log_message(f"Lambda NCE: {self.lambda_nce}")
        self.log_message(f"Lambda Identity: {self.lambda_identity}")

    def _initialize_weights(self, model):
        """Initialize network weights using Kaiming initialization"""
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                # Break line 234 (was ~217)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def add_noise_to_input(self, tensor, noise_factor=None):
        """Add Gaussian noise to input tensor for stability"""
        noise_level = noise_factor if noise_factor is not None else self.noise_factor
        if noise_level <= 0: return tensor
        return tensor + torch.randn_like(tensor) * noise_level

    def _get_features(self, x, get_intermediate=False):
        """
        Extract features from generator at multiple layers for NCE loss.
        If get_intermediate=True, return all intermediate features as well as final output.
        """
        # Initialize list to store features
        features = []

        # Get generator encoder features
        if hasattr(self.generator, 'get_features'):
            # For generators with built-in feature extraction
            all_features = self.generator.get_features(x)

            # Check if we need only selected layers
            if hasattr(self, 'nce_layers') and self.nce_layers:
                # Extract features from specified layers
                features = [all_features[i] for i in self.nce_layers]
            else:
                # Use all features
                features = all_features
        else:
            # Basic feature extraction - use the input itself
            features = [x]

        # Get final output from generator
        output = self.generator(x)

        if get_intermediate:
            # Return both intermediate features and final output
            return features, output
        else:
            # Return only intermediate features
            return features

    def init_patch_sampler(self):
        """Initialize the patch sampler optimizer and model"""
        try:
            # Create a dummy input to initialize features
            dummy_input = torch.zeros(1, 3, 256, 256, device=self.device)
            dummy_features = self._get_features(dummy_input)

            # Initialize patch sampler with these features
            if hasattr(self.patch_sampler, 'create_mlp_layers'):
                self.patch_sampler.create_mlp_layers(dummy_features)

            # Create optimizer for patch sampler
            self.patch_sampler_optimizer = torch.optim.Adam(
                self.patch_sampler.parameters(),
                lr=self.args.lr * 0.1,  # Lower learning rate
                betas=(self.args.b1, self.args.b2))  # Use b1 and b2
            self.log_message("Patch sampler initialized successfully")
        except Exception as e:
            self.log_message(f"Error initializing patch sampler: {e}")
            # Create a minimal optimizer with any available parameters
            params = [p for p in self.patch_sampler.parameters() if p.requires_grad]
            if len(params) > 0:
                self.patch_sampler_optimizer = torch.optim.Adam(params,
                                                                lr=self.args.lr * 0.1,
                                                                betas=(self.args.b1, self.args.b2))  # Use b1 and b2
                self.log_message(f"Created fallback optimizer")
            else:
                self.log_message("Creating dummy parameter")
                # Create dummy parameter if none exists
                if not hasattr(self.patch_sampler, 'dummy_param'):
                    self.patch_sampler.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True,
                                                                              device=self.device))
                self.patch_sampler_optimizer = torch.optim.Adam([self.patch_sampler.dummy_param],
                                                                lr=self.args.lr * 0.1,
                                                                betas=(self.args.b1, self.args.b2))  # Use b1 and b2

    def compute_nce_loss(self, real, fake):
        """
        Compute NCE (Noise Contrastive Estimation) loss between real and fake images
        using the patch sampler and PatchNCELoss.
        """
        # Get features for real and fake
        real_features, _ = self._get_features(real, get_intermediate=True)
        fake_features, _ = self._get_features(fake, get_intermediate=True)

        # Initialize NCE loss
        total_nce_loss = 0.0
        n_layers = 0

        # Compute NCE loss for corresponding layers
        for f_real, f_fake in zip(real_features, fake_features):
            # Sample patches
            feat_k_pool, sample_ids = self.patch_sampler([f_real], num_patches=256)
            feat_q_pool, _ = self.patch_sampler([f_fake], num_patches=256, patch_ids=sample_ids)

            # Compute NCE loss for this layer
            nce_loss = self.criterion_nce(feat_q_pool[0], feat_k_pool[0])
            total_nce_loss += nce_loss
            n_layers += 1

        # Average loss over all layers
        if n_layers > 0:
            total_nce_loss = total_nce_loss / n_layers

        # Ensure we return a tensor, not a float
        if not torch.is_tensor(total_nce_loss):
            total_nce_loss = torch.tensor(total_nce_loss, device=self.device, requires_grad=True)

        return total_nce_loss

    def compute_generator_loss(self, fake_outputs, fake_images, real_images):
        """
        Compute combined generator loss: GAN + NCE + Identity
        """
        # GAN loss - Make discriminator think generated images are real
        gan_loss = F.binary_cross_entropy_with_logits(fake_outputs, torch.ones_like(fake_outputs))

        # NCE loss - Contrastive learning between real and fake images
        nce_loss = self.compute_nce_loss(real_images, fake_images)

        # Identity loss - Try to reconstruct real image if given as input
        identity_loss = 0.0
        if self.lambda_identity > 0:
            identity = self.generator(real_images)
            identity_loss = F.l1_loss(identity, real_images)

        # Total loss
        total_loss = (self.lambda_gan * gan_loss + self.lambda_nce * nce_loss + self.lambda_identity * identity_loss)

        # Return loss components as floats to avoid tensor detachment issues
        return total_loss, {
            'gan': gan_loss.item(),
            'nce': nce_loss.item(),
            'identity': identity_loss.item() if torch.is_tensor(identity_loss) else identity_loss
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.generator.train()
        if self.discriminator:
            self.discriminator.train()
        if self.patch_sampler:
            self.patch_sampler.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        gen_loss = []
        dis_loss = []
        last_batch = None

        # Set up mixed precision training based on PyTorch version
        if torch.cuda.is_available():
            torch_version = pkg_resources.get_distribution("torch").version
            is_torch_2 = int(torch_version.split('.')[0]) >= 2

            if is_torch_2:
                from torch.amp import GradScaler
                from torch.amp.autocast_mode import autocast
                scaler_g = GradScaler()
                scaler_d = GradScaler()
                autocast_ctx = autocast(device_type='cuda')
            else:
                from torch.cuda.amp import GradScaler, autocast
                scaler_g = GradScaler()
                scaler_d = GradScaler()
                autocast_ctx = autocast()
        else:
            # CPU fallback
            scaler_g = None
            scaler_d = None
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        # Create progress bar
        if hasattr(self, 'rich_available') and self.rich_available:
            progress = self.create_rich_progress()
            if progress:
                task_id = progress.add_task(f"[cyan]Epoch {self.epoch + 1}/{self.args.epochs}",
                                            total=len(self.train_loader))
                progress.start()
        else:
            pbar = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader),
                        desc=f"Epoch {self.epoch + 1}/{self.args.epochs}",
                        bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        colour='green')

        batch_count = 0

        for i, (low_images, high_images) in enumerate(self.train_loader):
            batch_count += 1

            # Move data to device
            low_images = low_images.to(self.device)
            high_images = high_images.to(self.device)

            # Add noise for stability
            low_images = self.add_noise_to_input(low_images)
            high_images = self.add_noise_to_input(high_images)

            # Update discriminator if it exists
            if self.discriminator:
                # Zero gradients
                if self.d_optimizer is not None:
                    self.d_optimizer.zero_grad(set_to_none=True)

                # Mixed precision handling
                if scaler_d is not None:
                    with autocast_ctx:
                        # Generate fake images
                        with torch.no_grad():
                            fake_images = self.generator(low_images)

                        # Get discriminator outputs
                        real_outputs = self.discriminator(high_images)
                        fake_outputs = self.discriminator(fake_images.detach())

                        # Create labels with noise and label smoothing
                        real_label = torch.ones_like(real_outputs) * (1 - 0.1)
                        fake_label = torch.zeros_like(fake_outputs)
                        d_real = F.binary_cross_entropy_with_logits(real_outputs, real_label)
                        d_fake = F.binary_cross_entropy_with_logits(fake_outputs, fake_label)
                        d_loss = (d_real + d_fake) / 2

                        d_x = torch.sigmoid(real_outputs).mean().item()

                    # Ensure we have a valid loss before scaling
                    if torch.isfinite(d_loss).all():
                        scaler_d.scale(d_loss).backward()
                        scaler_d.unscale_(self.d_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_discriminator)
                        scaler_d.step(self.d_optimizer)
                        scaler_d.update()
                    else:
                        # Skip the step if loss is not finite
                        self.log_message(f"Skipping discriminator step due to non-finite loss: {d_loss.item()}")
                else:
                    # Non-mixed precision path
                    fake_images = self.generator(low_images)
                    real_outputs = self.discriminator(high_images)
                    fake_outputs = self.discriminator(fake_images.detach())

                    real_label = torch.ones_like(real_outputs) * (1 - 0.1)
                    fake_label = torch.zeros_like(fake_outputs)
                    d_real = F.binary_cross_entropy_with_logits(real_outputs, real_label)
                    d_fake = F.binary_cross_entropy_with_logits(fake_outputs, fake_label)
                    d_loss = (d_real + d_fake) / 2

                    d_x = torch.sigmoid(real_outputs).mean().item()

                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_discriminator)
                    self.d_optimizer.step()

                running_d_loss += d_loss.item()
                dis_loss.append(d_loss.item())

            # Update generator
            self.g_optimizer.zero_grad(set_to_none=True)

            # Mixed precision handling for generator
            if scaler_g is not None:
                with autocast_ctx:
                    # Forward pass
                    fake_images = self.generator(low_images)

                    # Calculate generator loss
                    if self.discriminator:
                        fake_outputs = self.discriminator(fake_images)
                        g_loss, loss_components = self.compute_generator_loss(fake_outputs, fake_images, high_images)
                    else:
                        g_loss = self.compute_nce_loss(high_images, fake_images)
                        loss_components = {'nce': g_loss.item() if torch.is_tensor(g_loss) else g_loss}

                # Ensure we have a valid loss before scaling
                if torch.is_tensor(g_loss) and torch.isfinite(g_loss).all():
                    scaler_g.scale(g_loss).backward()
                    scaler_g.unscale_(self.g_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_generator)
                    scaler_g.step(self.g_optimizer)

                    if self.patch_sampler_optimizer:
                        scaler_g.unscale_(self.patch_sampler_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.patch_sampler.parameters(), self.grad_clip_generator)
                        scaler_g.step(self.patch_sampler_optimizer)

                    scaler_g.update()
                else:
                    # Skip the step if loss is not finite
                    val = g_loss if not torch.is_tensor(g_loss) else g_loss.item()
                    self.log_message(f"Skipping generator step due to non-finite loss: {val}")
            else:
                # Non-mixed precision path
                fake_images = self.generator(low_images)

                if self.discriminator:
                    fake_outputs = self.discriminator(fake_images)
                    g_loss, loss_components = self.compute_generator_loss(fake_outputs, fake_images, high_images)
                else:
                    g_loss = self.compute_nce_loss(high_images, fake_images)
                    loss_components = {'nce': g_loss.item() if torch.is_tensor(g_loss) else g_loss}

                if torch.is_tensor(g_loss):
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_generator)
                    self.g_optimizer.step()

                    if self.patch_sampler_optimizer:
                        torch.nn.utils.clip_grad_norm_(self.patch_sampler.parameters(), self.grad_clip_generator)
                        self.patch_sampler_optimizer.step()

            if torch.is_tensor(g_loss):
                g_item = g_loss.item()
                running_g_loss += g_item
                gen_loss.append(g_item)
            else:
                running_g_loss += g_loss
                gen_loss.append(g_loss)

            # Save the last batch results for visualization
            last_batch = (low_images, high_images, fake_images)

            # Update progress information
            if batch_count % 10 == 0:
                lr_g = self.g_optimizer.param_groups[0]['lr']
                lr_d = (self.d_optimizer.param_groups[0]['lr'] if self.d_optimizer else 0.0)

                if d_loss is not None:
                    d_loss_info = f"D: {d_loss.item():.4f} | "
                else:
                    d_loss_info = ""

                progress_info = (f"G: {g_loss.item():.4f} | "
                                 f"{d_loss_info}"
                                 f"LR_G: {lr_g:.6f}")

                if self.d_optimizer:
                    progress_info += f" | LR_D: {lr_d:.6f}"

                if (hasattr(self, 'rich_available') and self.rich_available and 'progress' in locals()
                        and progress is not None and 'task_id' in locals()):
                    progress.update(task_id,
                                    advance=1,
                                    description=(f"[cyan]Epoch {self.epoch + 1}/{self.args.epochs}"
                                                 f" - {progress_info}"))
                elif 'pbar' in locals():
                    pbar.set_description(f"🔄 {progress_info}")

        # Close progress bar
        if (hasattr(self, 'rich_available') and self.rich_available and 'progress' in locals()
                and progress is not None):
            progress.stop()
        elif 'pbar' in locals():
            pbar.close()

        # Calculate average losses
        avg_g_loss = running_g_loss / len(self.train_loader)
        avg_d_loss = (running_d_loss / len(self.train_loader) if self.discriminator else 0.0)

        return avg_g_loss, avg_d_loss, last_batch

    def save_checkpoint(self, is_best=False):
        """Save the model checkpoints including the patch sampler"""
        # Create directories if they don't exist
        generator_dir = os.path.join(self.path, 'generator')
        discriminator_dir = os.path.join(self.path, 'discriminator')
        patch_sampler_dir = os.path.join(self.path, 'patch_sampler')

        os.makedirs(generator_dir, exist_ok=True)
        if self.discriminator:
            os.makedirs(discriminator_dir, exist_ok=True)
        os.makedirs(patch_sampler_dir, exist_ok=True)

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

        if self.patch_sampler_optimizer is not None:
            ps_checkpoint = {
                'net': self.patch_sampler.state_dict(),
                'optimizer': self.patch_sampler_optimizer.state_dict(),
                'epoch': self.epoch
            }

            torch.save(ps_checkpoint, ps_save_path)

        self.log_message(f"Saved checkpoint at epoch {self.epoch + 1}")

    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoints including the patch sampler"""
        # If no path provided, use the existing resume path
        checkpoint_path = checkpoint_path or self.args.resume

        if not checkpoint_path or checkpoint_path == '':
            self.log_message("No checkpoint path provided")
            return

        self.log_message(f"Loading checkpoint from {checkpoint_path}")

        # Load generator checkpoint
        g_path_checkpoint = os.path.join(checkpoint_path, 'generator/last.pt')

        if os.path.exists(g_path_checkpoint):
            try:
                g_checkpoint = torch.load(g_path_checkpoint, map_location=self.device, weights_only=True)
            except Exception:
                g_checkpoint = torch.load(g_path_checkpoint, map_location=self.device, weights_only=False)

            self.generator.load_state_dict(g_checkpoint['net'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            self.epoch = g_checkpoint['epoch']

            self.log_message(f"Loaded generator checkpoint")

        # Load discriminator checkpoint
        if self.discriminator:
            d_path_checkpoint = os.path.join(checkpoint_path, 'discriminator/last.pt')
            if os.path.exists(d_path_checkpoint):
                try:
                    d_checkpoint = torch.load(d_path_checkpoint, map_location=self.device, weights_only=True)
                except Exception:
                    d_checkpoint = torch.load(d_path_checkpoint, map_location=self.device, weights_only=False)

                self.discriminator.load_state_dict(d_checkpoint['net'])
                if self.d_optimizer:
                    self.d_optimizer.load_state_dict(d_checkpoint['optimizer'])

                self.log_message("Loaded discriminator checkpoint")

        # Load patch sampler checkpoint
        ps_path_checkpoint = os.path.join(checkpoint_path, 'patch_sampler/last.pt')
        if os.path.exists(ps_path_checkpoint):
            try:
                ps_checkpoint = torch.load(ps_path_checkpoint, map_location=self.device, weights_only=True)
            except Exception:
                ps_checkpoint = torch.load(ps_path_checkpoint, map_location=self.device, weights_only=False)

            self.patch_sampler.load_state_dict(ps_checkpoint['net'])
            if self.patch_sampler_optimizer:
                self.patch_sampler_optimizer.load_state_dict(ps_checkpoint['optimizer'])

            self.log_message("Loaded patch sampler checkpoint")

    def _apply_spectral_norm(self, model):
        """Apply spectral normalization to convolutional layers."""
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                spectral_norm(module)


def create_model(args):
    """Create generator and discriminator models based on args"""
    # Create the appropriate generator model
    if args.model_type.lower() == 'cut':
        generator = CUTGenerator(depth=1, weight=1)
    elif args.model_type.lower() == 'fastcut':
        generator = FastCUTGenerator(depth=0.75, weight=0.75)
    elif args.model_type.lower() == 'repvitcut':
        generator = RepVitCUTGenerator(depth=1, weight=1)
    elif args.model_type.lower() == 'repvitfastcut':
        generator = RepVitFastCUTGenerator(depth=0.75, weight=0.75)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Create discriminator if needed
    discriminator = Discriminator(depth=1, weight=1)

    return generator, discriminator


def train_cut(args, fast_mode=False):
    """
    Train a CUT/FastCUT model
    
    Args:
        args: Training arguments
        fast_mode: Whether to use FastCUT (fewer NCE losses)
    """
    # Set random seed for reproducibility
    set_random_seed(args.seed, args.deterministic, args.benchmark)

    # Create models
    generator, discriminator = create_model(args)

    # Determine model type based on args and fast_mode
    if args.model_type.lower() in ['cut', 'repvitcut'] and not fast_mode:
        model_type = "RepVitCUT" if args.model_type.lower() == 'repvitcut' else "CUT"
    elif args.model_type.lower() in ['fastcut', 'repvitfastcut'] or fast_mode:
        model_type = ("RepVitFastCUT" if args.model_type.lower() == 'repvitfastcut' else "FastCUT")
    else:
        model_type = "CUT"  # Default fallback

    print(f"Initializing {model_type} model")

    # Create trainer
    trainer = CUTTrainer(args, generator, discriminator, fast_mode=fast_mode)

    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_data = LowLightDataset(image_dir=args.data, transform=transform, phase="train")
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=True)

    test_data = LowLightDataset(image_dir=args.data, transform=test_transform, phase="test")
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             pin_memory=True)

    # Set train loader for trainer
    trainer.train_loader = train_loader
    trainer.test_loader = test_loader

    # Train the model
    trainer.train()

    return trainer


# Create helper functions to be called from DeepTranserGAN.py
def train_CUT(args):
    """Train a CUT model"""
    return train_cut(args, fast_mode=False)


def train_FastCUT(args):
    """Train a FastCUT model"""
    return train_cut(args, fast_mode=True)


# RepVit-Enhanced version of CUT generator
class RepVitCUTGenerator(CUTGenerator):
    """
    Enhanced CUT Generator using RepVit blocks for better feature representation.
    """

    def __init__(self, depth=1, weight=1):
        super().__init__(depth, weight)
        base_ch = math.ceil(8 * self.depth)

        # Enhance bottleneck with more RepVit blocks
        self.bottleneck = nn.Sequential(SPPELAN(base_ch * 8, base_ch * 8, base_ch * 4), PSA(base_ch * 8, base_ch * 8),
                                        RepViTBlock(base_ch * 8, base_ch * 8, 1, 1, 1, 1),
                                        RepViTBlock(base_ch * 8, base_ch * 8, 1, 1, 1, 1))

    def _make_downblock(self, in_ch, out_ch):
        """Enhanced downblock with SE attention"""
        return nn.Sequential(RepViTBlock(in_ch, out_ch, math.ceil(self.weight), 2, 1, 1),
                             RepViTBlock(out_ch, out_ch, 1, 1, 1, 1))

    def _make_upblock(self, in_ch, out_ch):
        """Enhanced upblock with SE attention"""
        return nn.Sequential(RepViTBlock(in_ch, in_ch, math.ceil(self.weight), 1, 1, 1),
                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), Gencov(in_ch, out_ch))


# RepVit-Enhanced version of FastCUT generator
class RepVitFastCUTGenerator(RepVitCUTGenerator):
    """
    Enhanced FastCUT Generator using RepVit blocks with reduced complexity.
    Inherits from RepVitCUTGenerator but uses fewer parameters.
    """

    def __init__(self, depth=0.75, weight=0.75):
        # Convert float to int to avoid type errors
        super().__init__(int(depth * 1.33), int(weight * 1.33))
        # Use smaller bottleneck and fewer layers for faster training


# Additional training functions
def train_RepVitCUT(args):
    """
    Train a CUT model with RepVit enhancements
    """
    # Override model type
    args.model_type = 'repvitcut'
    return train_cut(args, fast_mode=False)


def train_RepVitFastCUT(args):
    """
    Train a FastCUT model with RepVit enhancements
    """
    # Override model type
    args.model_type = 'repvitfastcut'
    return train_cut(args, fast_mode=True)


def parse_args():
    """Parse command line arguments for CUT training"""
    parser = argparse.ArgumentParser(description='DeepTranserCUT Training')

    # Model parameters
    parser.add_argument('--model-type',
                        type=str,
                        default='cut',
                        choices=['cut', 'fastcut', 'repvitcut', 'repvitfastcut'],
                        help='Model type: cut, fastcut, repvitcut, repvitfastcut')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                        help='Optimizer type')
    parser.add_argument('--loss',
                        type=str,
                        default='bce',
                        choices=['mse', 'bce', 'focalloss', 'bceblurwithlogitsloss'],
                        help='Loss function type')

    # Dataset parameters
    parser.add_argument('--data', type=str, default='../datasets/kitti_LOL', help='Dataset root directory')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=256, help='Size of input images')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=3.5e-4, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam optimizer beta1')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam optimizer beta2')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizers')
    parser.add_argument('--scheduler',
                        type=str,
                        default='warmup_cosine',
                        choices=['warmup_cosine', 'step', 'cosine', 'none'],
                        help='Learning rate scheduler type')

    # Loss weights and parameters
    parser.add_argument('--nce-temp', type=float, default=0.07, help='Temperature for NCE loss')
    parser.add_argument('--nce-weight', type=float, default=1.0, help='Weight for NCE loss')
    parser.add_argument('--identity-weight', type=float, default=0.1, help='Weight for identity loss')
    parser.add_argument('--gan-weight', type=float, default=1.0, help='Weight for GAN loss')
    parser.add_argument('--lambda-gp', type=float, default=10.0, help='Gradient penalty lambda for WGAN-GP')

    # Training stability
    parser.add_argument('--noise-factor', type=float, default=0.05, help='Noise factor for training stability')
    parser.add_argument('--grad-clip-gen', type=float, default=5.0, help='Gradient clipping value for generator')
    parser.add_argument('--grad-clip-disc', type=float, default=1.0, help='Gradient clipping value for discriminator')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')

    # Device and environment
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode')
    parser.add_argument('--benchmark', action='store_true', help='Enable cudnn benchmark')

    # Saving and loading
    parser.add_argument('--save-path', type=str, default='./runs/', help='Path to save models and logs')
    parser.add_argument('--resume', type=str, default='', help='Path to resume training from checkpoint')
    parser.add_argument('--save-freq', type=int, default=10, help='Frequency of saving checkpoints (epochs)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--min-improvement', type=float, default=1e-4, help='Minimum improvement for early stopping')

    # Logging
    parser.add_argument('--log-freq', type=int, default=100, help='Frequency of logging training status (steps)')
    parser.add_argument('--eval-freq', type=int, default=1, help='Frequency of evaluation (epochs)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_cut(args)


if __name__ == '__main__':
    main()
