import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.utils import make_grid

class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        lr=2e-5,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="online",
        conditional=True,
        pred_steps=None,
        pred_checkpoints=[],
        data_norm=True,
        temp=1.0,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        self.sample_from = sample_from
        self.conditional = conditional
        self.online_network = online_network
        self.target_network = target_network

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.data_norm = data_norm
        self.temp = temp

        # Disable automatic optimization
        self.automatic_optimization = False

    def forward(self, x, y,cond=None, n_steps=None, checkpoints=[]):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        return sample_nw.sample(x, y,cond=cond, n_steps=n_steps, checkpoints=checkpoints)

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        y=None
        if self.conditional:
            x, cond = batch
        else:
            y,x=batch
        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)   
        # Predict noise
        eps_pred = self.online_network(x, eps, t, y=y, cond=cond)

        # Compute loss
        loss = self.criterion(eps, eps_pred)
        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, cond = batch
        x_T = torch.randn_like(x)
        result = self(x_T, y=None, cond=cond, n_steps=1000)["1000"]
        self.logger.experiment.add_image("results", result[0], self.current_epoch)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            x_t = batch
            return self(
                x_t,
                cond=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
            )

        (recons, _), x_t = batch
        x_t = self.temp * x_t[0]  # This is really a one element tuple

   
        return (
            self(
                x_t,
                cond=recons,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
            ),
            recons,
        )
    def load_model(self):
        return self.target_network.decoder

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
