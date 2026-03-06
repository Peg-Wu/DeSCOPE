import os
import torch
import inspect
import logging
from tqdm.auto import tqdm
from wppkg import (
    Trainer, 
    Accumulator, 
    get_logger
)


def _init_logger(self) -> logging.Logger:
    # Create an independent logger for the Trainer.
    log_file = os.path.join(self.args.output_dir, "run.log")
    logger = get_logger(
        name="wppkg.Trainer",
        log_file=log_file,
        log_file_mode="w",
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        main_process_level=logging.INFO,
        other_process_level=logging.WARN,
        local_rank=self.accelerator.local_process_index
    )
    # Silence Console Output
    logger.info(f"Training logs have been written to {os.path.abspath(log_file)}")
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.ERROR)
    return logger


Trainer._init_logger = _init_logger


class DeSCOPETrainer(Trainer):
    r"""
    NOTE: 
        1. Early stopping does not currently support resuming training. 
           If training is forcibly resumed, the early stopping callback will be reinitialized.
        2. If you enable early stopping, ensure that `eval_every_n_epochs` and `checkpointing_steps` are aligned, 
           as the Trainer does not automatically save the best model.
        3. The final model is always saved at the end of training, even if early stopping is triggered.
    """
    def train(self):
        # Train!
        total_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.logger.info("*****************************")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint is not None or self.args.resume_from_checkpoint != "":
                checkpoint_path = self.args.resume_from_checkpoint
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * self.num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * self.args.gradient_accumulation_steps
                starting_epoch = resume_step // len(self.train_dataloader)
                completed_steps = resume_step // self.args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(self.train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        accumulator_train = Accumulator(name=["loss", "mse_loss", "kl_loss"])
        model_forward_keys = list(inspect.signature(self.model.forward).parameters.keys())
        for epoch in range(starting_epoch, self.args.num_train_epochs):
            self.model.train()
        
            if self.args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
            else:
                active_dataloader = self.train_dataloader
            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    filtered_batch = {k: v.to(self.accelerator.device) for k, v in batch.items() if k in model_forward_keys}
                    outputs = self.model(**filtered_batch)
                    loss = outputs.loss
                    mse_loss = outputs.mse_loss
                    kl_loss = outputs.kl_loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                
                # We keep track of the loss at each logging_steps
                accumulator_train.add(
                    self.accelerator.reduce(loss.detach().clone(), "mean").item(),
                    self.accelerator.reduce(mse_loss.detach().clone(), "mean").item(),
                    self.accelerator.reduce(kl_loss.detach().clone(), "mean").item()
                )
                
                # Log training progress
                if completed_steps % self.args.logging_steps == 0:
                    accumulator_train.mean()
                    log_dict = accumulator_train.to_dict()
                    accumulator_train.reset()  # reset accumulator
                    extra_log_dict = {
                        "grad_norm": grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm,
                        "lr": self.lr_scheduler.get_last_lr()[0]
                    }
                    log_dict = log_dict | extra_log_dict
                    log_dict_round = {
                        k: round(v, 6) if k == "lr" else round(v, 4)
                        for k, v in log_dict.items()
                    }
                    self.logger.info({"epoch": epoch, "step": completed_steps, **log_dict_round})

                    if self.args.with_tracking:
                        self.accelerator.log(log_dict, step=completed_steps)

                if isinstance(self.args.checkpointing_steps, int):
                    if completed_steps % self.args.checkpointing_steps == 0 and self.accelerator.sync_gradients:
                        output_dir = f"step_{completed_steps}"
                        output_dir = os.path.join(self.args.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)
                        # Save the model checkpoint et al.
                        self._save(os.path.join(output_dir, "model"))

                if completed_steps >= self.args.max_train_steps:
                    break
            
            # NOTE: Evaluation will be performed at the end of each epoch. (or every `eval_every_n_epochs`)
            if self.eval_dataloader is not None and (epoch + 1) % self.args.eval_every_n_epochs == 0:
                eval_log_dict = self.evaluate()

                # Log evaluation progress
                self.logger.info({"epoch": epoch, **eval_log_dict})
                if self.args.with_tracking:
                    self.accelerator.log(eval_log_dict, step=epoch)
                
                # EarlyStop: check if we should stop the training on any processes
                if self.earlystop_callback is not None:
                    if self.earlystop_callback.check_early_stopping(eval_log_dict["eval_loss"]):
                        self.accelerator.set_trigger()
                    # If so, we break the loop
                    if self.accelerator.check_trigger():
                        self.logger.info(f"Model has not improved for {self.args.earlystop_patience} evaluations, so we halt the training session.")
                        break
            
            # NOTE: Allow checkpointing_steps to be in the format "epoch-<number>", meaning a checkpoint is saved every <number> epochs.
            if isinstance(self.args.checkpointing_steps, str):
                checkpointing_every_n_epochs = (
                    1 
                    if self.args.checkpointing_steps == "epoch" 
                    else int(self.args.checkpointing_steps.split("-")[-1])
                )

                if (epoch + 1) % checkpointing_every_n_epochs == 0:
                    output_dir = f"epoch_{epoch}"
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                    self.accelerator.save_state(output_dir)
                    # Save the model checkpoint et al.
                    self._save(os.path.join(output_dir, "model"))

        # Save the last model checkpoint.
        self._save(os.path.join(self.args.output_dir, "last_model"))
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        self.logger.info("Training exited successfully.")
    
    def evaluate(self):
        self.model.eval()
        losses, mse_losses, kl_losses = [], [], []
        model_forward_keys = list(inspect.signature(self.model.forward).parameters.keys())
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                filtered_batch = {k: v.to(self.accelerator.device) for k, v in batch.items() if k in model_forward_keys}
                outputs = self.model(**filtered_batch)
            
            loss, mse_loss, kl_loss = outputs.loss, outputs.mse_loss, outputs.kl_loss
            losses.append(self.accelerator.gather_for_metrics(loss.repeat(self.args.per_device_eval_batch_size)))
            mse_losses.append(self.accelerator.gather_for_metrics(mse_loss.repeat(self.args.per_device_eval_batch_size)))
            kl_losses.append(self.accelerator.gather_for_metrics(kl_loss.repeat(self.args.per_device_eval_batch_size)))

        eval_loss = torch.mean(torch.cat(losses))
        eval_mse_loss = torch.mean(torch.cat(mse_losses))
        eval_kl_loss = torch.mean(torch.cat(kl_losses))

        return {
            "eval_loss": eval_loss.item(),
            "eval_mse_loss": eval_mse_loss.item(),
            "eval_kl_loss": eval_kl_loss.item()
        }
