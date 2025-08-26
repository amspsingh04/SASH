import os
import time
import copy
import numpy as np
import torch
import torchaudio
from glob import glob
import re
import hydra
import wandb
import omegaconf

from utils.torch_utils import training_stats
from utils.torch_utils import misc
import utils.log as utils_logging
import utils.training_utils as t_utils

#----------------------------------------------------------------------------

class Trainer():
    def __init__(self, args=None, dset=None, network=None, diff_params=None, tester=None, device='cpu'):

        assert args is not None, "args dictionary is None"
        self.args=args

        assert dset is not None, "dset is None"
        self.dset=dset

        assert network is not None, "network is None"
        self.network=network

        assert diff_params is not None, "diff_params is None"
        self.diff_params=diff_params

        assert device is not None, "device is None"
        self.device=device

        self.tester = tester
        if self.tester is not None:
            self.tester.use_wandb = False # We do not want to interfere with the training wandb, as we do the logging in Trainer() and not in Tester()

        self.optimizer = hydra.utils.instantiate(args.exp.optimizer, params=network.parameters())

        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)

        # Torch settings
        torch.manual_seed(self.args.exp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        self.total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print("total_params: ",self.total_params/1e6, "M")

        # Checkpoint Resuming
        self.latest_checkpoint = None
        resuming = False
        if self.args.exp.resume:
            if self.args.exp.resume_checkpoint != "None":
                resuming = self.resume_from_checkpoint(checkpoint_path=self.args.exp.resume_checkpoint)
            else:
                resuming = self.resume_from_checkpoint()
            if not resuming:
                print("Could not resume from checkpoint")
                print("training from scratch")
            else:
                print("Resuming from iteration {}".format(self.it))
        if not resuming:
            self.it = 0
            self.latest_checkpoint = None
            if tester is not None:
                self.tester.it = 0

        # Model Summary
        if self.args.logging.print_model_summary:
            with torch.no_grad():
                audio = torch.zeros([args.exp.batch_size,args.exp.audio_len], device=device).unsqueeze(1)
                sigma = torch.ones([args.exp.batch_size], device=device)
                misc.print_module_summary(self.network, [audio, sigma ], max_nesting=2)

        # Logger Setup
        if self.args.logging.log:
            self.setup_wandb()
            self.setup_logging_variables()

        # Profiler
        self.profiler, self.profile, self.profile_total_steps = t_utils.profile(self.args.logging)

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        config["total_params"]=self.total_params
        self.wandb_run=wandb.init(project=self.args.logging.wandb.project, config=config, dir=self.args.model_dir)
        wandb.watch(self.network, log="all", log_freq=self.args.logging.heavy_log_interval) #wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name = os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id #adding the experiment number to the run name, bery important, I hope this does not crash

    def setup_logging_variables(self):
        self.sigma_bins = np.logspace(np.log10(self.args.diff_params.sde_hp.sigma_min), np.log10(self.args.diff_params.sde_hp.sigma_max), num=self.args.logging.num_sigma_bins, base=10)

    def load_state_dict(self, state_dict):
        return t_utils.load_state_dict(state_dict, network=self.network, ema=self.ema, optimizer=self.optimizer)

    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        # Resume training from latest checkpoint available in the output director
        if checkpoint_path is not None:
            try:
                # --- THIS LINE HAS BEEN MODIFIED ---
                checkpoint=torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=157007 #large number to mean that we loaded somethin, but it is arbitrary
                return self.load_state_dict(checkpoint)
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it=0

            try:
                # --- THIS LINE HAS BEEN MODIFIED ---
                checkpoint=torch.load(os.path.join(self.args.model_dir,checkpoint_path), map_location=self.device, weights_only=False)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=157007 #large number to mean that we loaded somethin, but it is arbitrary
                self.network.load_state_dict(checkpoint['ema_model'])
                return True
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it=0
                return False
        else:
            try:
                print("trying to load a project checkpoint")
                print("checkpoint_id", checkpoint_id)
                print("model_dir", self.args.model_dir)
                print("exp_name", self.args.exp.exp_name)
                if checkpoint_id is None:
                    # find latest checkpoint_id
                    save_basename = f"{self.args.exp.exp_name}-*.pt"
                    save_name = f"{self.args.model_dir}/{save_basename}"
                    list_weights = glob(save_name)
                    id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
                    list_ids = [int(id_regex.search(weight_path).groups()[0])
                                for weight_path in list_weights]
                    checkpoint_id = max(list_ids)

                # --- THIS LINE HAS BEEN MODIFIED ---
                checkpoint = torch.load(
                    f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device, weights_only=False)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=159000 #large number to mean that we loaded somethin, but it is arbitrary
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                print(e)
                return False

    def state_dict(self):
        return {
            'it': self.it,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'args': self.args,
        }

    def save_checkpoint(self):
        save_basename = f"{self.args.exp.exp_name}-{self.it}.pt"
        save_name = f"{self.args.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)
        print("saving",save_name)
        if self.args.logging.remove_old_checkpoints:
            try:
                os.remove(self.latest_checkpoint)
                print("removed last checkpoint", self.latest_checkpoint)
            except:
                print("could not remove last checkpoint", self.latest_checkpoint)
        self.latest_checkpoint=save_name


    def process_loss_for_logging(self, error: torch.Tensor, sigma: torch.Tensor):
        torch.nan_to_num(error) #not tested might crash
        error = error.detach().cpu().numpy()
        training_stats.report('loss', error.mean())

        for i in range(len(self.sigma_bins)):
            if i == 0:
                mask = sigma <= self.sigma_bins[i]
            elif i == len(self.sigma_bins)-1:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])
            else:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])
            mask = mask.squeeze(-1).cpu()
            if mask.sum() > 0:
                idx = np.where(mask==True)[0][0]
                training_stats.report('error_sigma_'+str(self.sigma_bins[i]),error[idx].mean())

    def get_batch(self):
        batch_dict = next(self.dset)
        sample = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch_dict.items()}
        return sample

    def train_step(self):
        self.optimizer.zero_grad()
        sample_dict = self.get_batch()
        clean_sample = sample_dict['reference']
        noise = None
        error, sigma = self.diff_params.loss_fn(self.network, clean_sample, n=noise)
        loss = error.mean()
        loss.backward()
        if self.args.exp.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)
        self.optimizer.step()
        if self.args.logging.log:
            self.process_loss_for_logging(error, sigma)

    def update_ema(self):
        ema_rampup = self.args.exp.ema_rampup
        ema_rate = self.args.exp.ema_rate
        t = self.it * self.args.exp.batch_size
        with torch.no_grad():
            if t < ema_rampup:
                s = np.clip(t / ema_rampup, 0.0, ema_rate)
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * s + src * (1-s))
            else:
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * ema_rate + src * (1-ema_rate))

    def easy_logging(self):
        training_stats.default_collector.update()
        loss_mean = training_stats.default_collector.mean('loss')
        self.wandb_run.log({'loss': loss_mean}, step=self.it)
        sigma_means, sigma_stds = [], []
        for i in range(len(self.sigma_bins)):
            sigma_means.append(training_stats.default_collector.mean('error_sigma_'+str(self.sigma_bins[i])))
            sigma_stds.append(training_stats.default_collector.std('error_sigma_'+str(self.sigma_bins[i])))
        figure = utils_logging.plot_loss_by_sigma(sigma_means,sigma_stds, self.sigma_bins)
        wandb.log({"loss_dependent_on_sigma": figure}, step=self.it, commit=True)

    def heavy_logging(self):
        if self.tester is not None:
            if self.latest_checkpoint is not None:
                self.tester.load_checkpoint(self.latest_checkpoint)
            audio = self.tester.do_test(it=self.it)
            for i, x in enumerate(audio):
                self.log_audio(x, f"sample_{i}")

    def log_audio(self, x, name):
        string = name+"_"+self.args.tester.name
        audio_path = utils_logging.write_audio_file(x,self.args.exp.sample_rate, string,path=self.args.model_dir, normalize=True)
        self.wandb_run.log({"audio_"+str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},step=self.it)
        if self.args.logging.log_spectrograms:
            spec_sample = utils_logging.plot_spectrogram_from_raw_audio(x, self.args.logging.stft)
            self.wandb_run.log({"spec_"+str(string): spec_sample}, step=self.it)

    def training_loop(self):
        while True:
            self.train_step()
            self.update_ema()
            if self.profile and self.args.logging.log:
                if self.it < self.profile_total_steps:
                    self.profiler.step()
                elif self.it == self.profile_total_steps +1:
                    profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
                    profile_art.add_file(glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
                    wandb.log_artifact(profile_art)
                    print("profiling done")
                elif self.it > self.profile_total_steps +1:
                    self.profile = False
            if self.it>0 and self.it%self.args.logging.save_interval==0 and self.args.logging.save_model:
                self.save_checkpoint()
            if self.it>0 and self.it%self.args.logging.heavy_log_interval==0 and self.args.logging.log:
                self.heavy_logging()
            if self.it>0 and self.it%self.args.logging.log_interval==0 and self.args.logging.log:
                self.easy_logging()
            self.it += 1
            try:
                if self.it > self.args.exp.max_iters:
                    break
            except:
                pass

