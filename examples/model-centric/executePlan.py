import warnings
warnings.filterwarnings("ignore")

import torch as th
from torchvision import datasets, transforms

import numpy as np
import urllib3
import time

import syft as sy
from syft.federated.fl_client import FLClient
from syft.federated.fl_job import FLJob
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient

urllib3.disable_warnings()
sy.make_hook(globals())

cycles_log = []
status = {
    "ended": False
}

# Called when client is accepted into FL cycle
def on_accepted(job: FLJob):
    print(f"Accepted into cycle {len(cycles_log) + 1}!")

    cycle_params = job.client_config
    batch_size = cycle_params["batch_size"]
    lr = cycle_params["lr"]
    max_updates = cycle_params["max_updates"]

    mnist_dataset = th.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )

    training_plan = job.plans["training_plan"]
    model_params = job.model.tensors()
    losses = []
    accuracies = []

    for batch_idx, (X, y) in enumerate(mnist_dataset):
        X = X.view(batch_size, -1)
        y_oh = th.nn.functional.one_hot(y, 10)
        loss, acc, *model_params = training_plan.torchscript(
            X, y_oh, th.tensor(batch_size), th.tensor(lr), model_params
        )
        losses.append(loss.item())
        accuracies.append(acc.item())
        if batch_idx % 50 == 0:
            print("Batch %d, loss: %f, accuracy: %f" % (batch_idx, loss, acc))
        if batch_idx >= max_updates:
            break

    job.report(model_params)
    # Save losses/accuracies from cycle
    cycles_log.append((losses, accuracies))

# Called when the client is rejected from cycle
def on_rejected(job: FLJob, timeout):
    if timeout is None:
        print(f"Rejected from cycle without timeout (this means FL training is done)")
    else:
        print(f"Rejected from cycle with timeout: {timeout}")
    status["ended"] = True

# Called when error occured
def on_error(job: FLJob, error: Exception):
    print(f"Error: {error}")
    status["ended"] = True

# PyGrid Node address
gridAddress = "http://alice:5000"

# Hosted model name/version
model_name = "mnist"
model_version = "1.0"

def new_job(self, model_name, model_version) -> FLJob:
        if self.worker_id is None:
            auth_response = self.grid_worker.authenticate(
                self.auth_token, model_name, model_version
            )
            self.worker_id = auth_response["data"]["worker_id"]

        job = FLJob(
            fl_client=self,
            grid_worker=self.grid_worker,
            model_name=model_name,
            model_version=model_version,
        )
        return job

def create_client_and_run_cycle():
    client = FLClient(url=gridAddress, auth_token=None, verbose=True)
    client.worker_id = client.grid_worker.authenticate(client.auth_token,model_name,model_version)["data"]["worker_id"]
    job = client.new_job(model_name, model_version)

    # Set event handlers
    job.add_listener(job.EVENT_ACCEPTED, on_accepted)
    job.add_listener(job.EVENT_REJECTED, on_rejected)
    job.add_listener(job.EVENT_ERROR, on_error)

    # Shoot!
    job.start()

while not status["ended"]:
    create_client_and_run_cycle()
    time.sleep(1)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, figsize=(10, 10))
axs[0].set_title("Loss")
axs[1].set_title("Accuracy")
offset = 0
for i, cycle_log in enumerate(cycles_log):
    losses, accuracies = cycle_log
    x = range(offset, offset + len(losses))
    axs[0].plot(x, losses)
    axs[1].plot(x, accuracies)
    fig.save('Model Loss and Accuracy.png')
    offset += len(losses)
    print(f"Cycle {i + 1}:\tLoss: {np.mean(losses)}\tAcc: {np.mean(accuracies)}")