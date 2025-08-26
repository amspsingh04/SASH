from datetime import date
import re
import torch
import torchaudio
import os
import numpy as np
import wandb
import copy
from glob import glob
from tqdm import tqdm
import omegaconf
import hydra
import soundfile as sf

from testing.operators.subband_filtering import BlindSubbandFiltering
from testing.operators.reverb import RIROperator

import utils.log as utils_logging
import utils.training_utils as tr_utils

class Tester():
    def __init__(
        self, args, network, diff_params, test_set=None, device=None, in_training=False,
    ):
        self.args=args
        self.network=network
        self.diff_params=copy.copy(diff_params)
        self.device=device
        self.test_set = test_set
        self.in_training = in_training

        self.sampler=hydra.utils.instantiate(args.tester.sampler, self.network, self.diff_params, self.args)

    def load_latest_checkpoint(self):
        #load the latest checkpoint from self.args.model_dir
        try:
            # find latest checkpoint_id
            save_basename = f"{self.args.exp.exp_name}-*.pt"
            save_name = f"{self.args.model_dir}/{save_basename}"
            list_weights = glob(save_name)
            id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
            list_ids = [int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights]
            checkpoint_id = max(list_ids)

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device, weights_only=False)
            try:
                self.network.load_state_dict(state_dict['ema'])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict['model'], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device,weights_only=False)
        try:
            self.it=state_dict['it']
        except:
            self.it=0
        print("loading checkpoint")
        return tr_utils.load_state_dict(state_dict, ema=self.network)

    def test_dereverberation(self, mode, blind=False):

        if self.test_set is None:
            print("No test set specified")
            return
        if len(self.test_set) == 0:
            print("No samples found in test set")
            return

        # --- THIS IS THE MAIN MODIFICATION ---
        # The loop now iterates through our dataloader which returns a dictionary.
        for i, batch_dict in enumerate(tqdm(self.test_set)):

            # Extract data from the dictionary
            seg = torch.from_numpy(batch_dict['reference']).float().to(self.device)   # Clean audio 
            y = torch.from_numpy(batch_dict['degraded']).float().to(self.device)     # Reverberant audio 
            filename = batch_dict['filename']

            # Normalize the input to match the model's expected data range
            y = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * y / y.std()

            with torch.no_grad():
                operator_blind = None
                if blind: # Initialize the blind operator for dereverberation
                    assert self.args.tester.blind_dereverberation.operator == "subband_filtering"
                    operator_blind = BlindSubbandFiltering(self.args.tester.informed_dereverberation.op_hp, sample_rate=self.args.exp.sample_rate)
                    with torch.no_grad():
                        operator_blind.update_H(use_noise=True)

            # Perform the prediction using the reverberant audio 'y'
            pred = self.sampler.predict_conditional(y.unsqueeze(0), operator_blind, shape=(1,seg.shape[-1]), blind=blind)

            # Save the results
            path_original=utils_logging.write_audio_file(seg, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"original"])
            path_degraded=utils_logging.write_audio_file(y, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"])
            path_reconstructed=utils_logging.write_audio_file(pred, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"reconstructed"])

            if blind and hasattr(self.sampler, 'operator'):
                utils_logging.write_audio_file(self.sampler.operator.get_time_RIR().detach().cpu(), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"estimated_rir"])

            print(f"Saved reconstructed file to: {path_reconstructed}")

    def prepare_directories(self, mode, unconditional=False, blind=False):
        today=date.today()
        self.paths={}

        if "overriden_name" in self.args.tester.keys() and self.args.tester.overriden_name is not None:
            self.path_sampling = os.path.join(self.args.model_dir, self.args.tester.overriden_name)
        else:
            self.path_sampling = os.path.join(self.args.model_dir,'test'+today.strftime("%d_%m_%Y"))
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        self.paths[mode]=os.path.join(self.path_sampling,mode,self.args.exp.exp_name)

        if not os.path.exists(self.paths[mode]):
            os.makedirs(self.paths[mode])

        if not unconditional:
            self.paths[mode+"original"]=os.path.join(self.paths[mode],"original")
            if not os.path.exists(self.paths[mode+"original"]):
                os.makedirs(self.paths[mode+"original"])
            self.paths[mode+"degraded"]=os.path.join(self.paths[mode],"degraded")
            if not os.path.exists(self.paths[mode+"degraded"]):
                os.makedirs(self.paths[mode+"degraded"])
            self.paths[mode+"reconstructed"]=os.path.join(self.paths[mode],"reconstructed")
            if not os.path.exists(self.paths[mode+"reconstructed"]):
                os.makedirs(self.paths[mode+"reconstructed"])

            if "dereverberation" in mode:
                self.paths[mode+"true_rir"]=os.path.join(self.paths[mode],"true_rir")
                if not os.path.exists(self.paths[mode+"true_rir"]):
                    os.makedirs(self.paths[mode+"true_rir"])

                if mode == "blind_dereverberation":
                    self.paths[mode+"estimated_rir"]=os.path.join(self.paths[mode],"estimated_rir")
                    if not os.path.exists(self.paths[mode+"estimated_rir"]):
                        os.makedirs(self.paths[mode+"estimated_rir"])

    def save_experiment_args(self, mode):
        with open(os.path.join(self.paths[mode], ".argv"), 'w') as f: #Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):
        self.it = it
        for m in self.args.tester.modes:
            if m == "blind_dereverberation":
                print("testing blind dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_dereverberation(m, blind=True)
            else:
                print("Warning: skipping unknown or unsupported mode: ", m)
