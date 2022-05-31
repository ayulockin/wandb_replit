import torch
import wandb
from rich import print
from rich.progress import Progress

def train(dataloader, model, loss_fn, optimizer, progress):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = size//num_batches

    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (pred > 0.5).float()
        train_acc += (pred == y).type(torch.float).sum().item()

        if batch % 10 == 0:
            tmp_loss, current = loss.item(), batch * len(X)
            # print(f"loss: {tmp_loss:>7f}  [{current:>5d}/{size:>5d}]")

        if wandb.run is not None:
            wandb.log({
                'batch/train_loss': loss.item(),
            })

        progress.advance(0, advance=batch_size)

    train_loss /= num_batches
    train_acc /= size
    # print(
    #     f"[bold magenta]Train Error[/bold magenta]: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n"
    # )

    if wandb.run is not None:
        wandb.log({
            'epoch/train_loss': train_loss,
            'epoch/train_acc': 100 * train_acc
        })


def test(dataloader, model, loss_fn, progress):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = size//num_batches

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            if pred <= 0.5:
                pred = 0
            else:
                pred = 1
            test_acc += (pred == y).type(torch.float).sum().item()

            progress.advance(1, advance=batch_size)

    test_loss /= num_batches
    test_acc /= size
    # print(
    #     f"[bold magenta]Test Error:[/bold magenta] \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    # )
    if wandb.run is not None:
        wandb.log({
            'epoch/test_loss': test_loss,
            'epoch/test_acc': 100 * test_acc
        })