import os

os.environ['WANDB_SILENT'] = "true"
import wandb
from argparse import Namespace
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader

from rich import print
from rich.live import Live
from rich.panel import Panel
from rich.progress import (Progress, SpinnerColumn, 
                           BarColumn, TextColumn, TimeElapsedColumn)
from rich.layout import Layout
from rich.table import Table, Column
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# Modules
from data import make_circle_dataset
from models import SimpleMLPModel
from pipeline import train, test

configs = Namespace(
    epochs=5,
    batch_size=8,
    model_hidden_size=100,
)


def setup_pipeline(configs):
    # Prepare datasets and dataloaders
    X_train, y_train = make_circle_dataset(n_samples=10000,
                                           factor=0.4,
                                           noise=0.1)
    X_test, y_test = make_circle_dataset(n_samples=1000, factor=0.5, noise=0.1)

    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    trainloader = DataLoader(train_dataset,
                             batch_size=configs.batch_size,
                             shuffle=True,
                             drop_last=True)
    testloader = DataLoader(test_dataset)

    # Build Model
    model = SimpleMLPModel(hidden_size=configs.model_hidden_size)

    # Loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return trainloader, testloader, model, criterion, optimizer


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="epoch-level", size=5),
        Layout(name="batch-level", size=8),
        Layout(name="footer", size=3),
    )

    return layout


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            "[b]1. :star: Experiment Tracking with Weights and Biases[/b] :bee:",
            datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(grid, style="white on black")


def get_epoch_table(epoch_progress):
    epoch_table = Table.grid(expand=True)
    epoch_table.add_row(
        Panel.fit(
            epoch_progress, title="Epoch Wise", border_style="green", padding=(1, 1)
        )
    )

    return epoch_table


def get_batch_table(batch_progress):
    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(batch_progress, title="[b]Batch Wise", border_style="red", padding=(2, 2))
    )

    return progress_table


def get_footer_table(url=None):
    footer_table = Table.grid(expand=True)
    footer_table.add_column(justify="center", ratio=1)
    footer_table.add_row(f"Check your progress at :point_right: [b]{url}[/b] :point_left:")
    return Panel(footer_table, style="white on black")


def ui(epoch_progress, batch_progress, url="Let W&B initialize a run!"):
    epoch_progress = get_epoch_table(epoch_progress)
    progress_table = get_batch_table(batch_progress)
    footer_table = get_footer_table(url)

    layout = make_layout()
    layout["header"].update(Header())
    layout["epoch-level"].update(epoch_progress)
    layout["batch-level"].update(progress_table)
    layout["footer"].update(footer_table)

    return layout


if __name__ == '__main__':
    trainloader, testloader, model, criterion, optimizer = setup_pipeline(configs)
    train_size, test_size = len(trainloader.dataset), len(testloader.dataset)
    train_num_batch, test_num_batch = len(trainloader), len(testloader)
    train_batch_size, test_batch_size = train_size//train_num_batch, test_size//test_num_batch

    # Setup epoch level progress bar.
    bar_column = BarColumn(bar_width=None, table_column=Column(ratio=1))
    epoch_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        bar_column,
        TimeElapsedColumn(),
        expand=True
    )
    epoch_task = epoch_progress.add_task("Num Epochs", total=configs.epochs, start=False)

    # Setup batch level progress bars.
    batch_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        bar_column,
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        expand=True)
    train_batch_task = batch_progress.add_task("[cyan]Train Batch", total=train_size*configs.epochs)
    test_batch_task = batch_progress.add_task("[cyan]Test Batch", total=test_size*configs.epochs)

    # Initialize the layout and train the model.
    layout = ui(epoch_progress, batch_progress)
    with Live(layout, refresh_per_second=10) as live:
        # Initialize W&B run
        run = wandb.init(project='pytorch-dropout', anonymous='must')

        live.update(ui(epoch_progress, batch_progress, run.get_url()))

        # Train and evaluate
        epoch_progress.start_task(0)
        for epoch in range(configs.epochs):    
            train(trainloader, model, criterion, optimizer, batch_progress)
            test(testloader, model, criterion, batch_progress)
            epoch_progress.update(epoch_task, advance=1)