from cmath import exp
import os
os.environ['WANDB_SILENT'] = "true"

import time
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
from rich.style import Style
from rich.prompt import Prompt, FloatPrompt, Confirm

# URL Shortner
import pyshorteners
type_tiny = pyshorteners.Shortener()

# Modules
from data import make_circle_dataset
from models import SimpleMLPModel
from pipeline import train, test

configs = Namespace(
    epochs=5,
    batch_size=8,
    model_hidden_size=100,
)

welcome = '''
    ╦ ╦┌─┐┬  ┌─┐┌─┐┌┬┐┌─┐  ┌┬┐┌─┐  ┌┬┐┬ ┬┌─┐        
   ║║║├┤ │  │  │ ││││├┤    │ │ │   │ ├─┤├┤         
    ╚╩╝└─┘┴─┘└─┘└─┘┴ ┴└─┘   ┴ └─┘   ┴ ┴ ┴└─┘        
                                                
                                                
                                                
╦ ╦┌─┐┬┌─┐┬ ┬┌┬┐┌─┐  ┌─┐┌┐┌┌┬┐  ╔╗ ┬┌─┐┌─┐┌─┐┌─┐
║║║├┤ ││ ┬├─┤ │ └─┐  ├─┤│││ ││  ╠╩╗│├─┤└─┐├┤ └─┐
╚╩╝└─┘┴└─┘┴ ┴ ┴ └─┘  ┴ ┴┘└┘─┴┘  ╚═╝┴┴ ┴└─┘└─┘└─┘
                                                
                                                
                                                
╔╦╗┬ ┬┌┬┐┌─┐┬─┐┬┌─┐┬  ┬                         
 ║ │ │ │ │ │├┬┘│├─┤│  │                         
 ╩ └─┘ ┴ └─┘┴└─┴┴ ┴┴─┘o                         

 '''


def setup_data_pipeline(configs):
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

    return trainloader, testloader


def make_welcome_layout(visible) -> Layout:
    """Define the layout."""
    layout = Layout(name="welcome")

    layout.split(
        Layout(name="welcome-screen", ratio=1)
    )

    if not visible:
        layout.visible = False

    return layout


def make_trainer_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="trainer")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="epoch-level", size=5),
        Layout(name="batch-level", size=8),
        Layout(name="footer", size=3),
    )

    return layout


class Welcome:
    """Display welcome message and ask for prompt."""

    def __rich__(self) -> Panel:
        grid_welcome = Table.grid(expand=True)
        grid_welcome.add_column(justify="center", ratio=1)
        grid_welcome.add_row(welcome)
        
        grid_intro = Table.grid(expand=True)
        grid_intro.add_column(justify="left", ratio=1, style="magenta")
        intro_text = Text("🐝 In this tutorial we will train a simple PyTorch model "
                           "on a toy dataset with different values of Dropout. While doing "
                           "so we will see how Weights and Biases can help compare different experiments. 🐝",
                           overflow="fold")
        grid_intro.add_row(intro_text)

        panel_group = Group(
            Panel(grid_welcome, style="white on blue"),
            Panel(grid_intro, style="white on red"),
        )

        custom_style = Style(color="cyan", blink=True, bold=True)
        return panel_group


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
    footer_table.add_column(justify="center", ratio=1, overflow='ellipsis')
    footer_table.add_row(f"Check your W&B run at :point_right: [b]{url}[/b] :point_left:")
    return Panel(footer_table, style="white on black")


def get_progress_bars():
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

    return epoch_progress, batch_progress


def welcome_ui(visible=True):
    layout = make_welcome_layout(visible=True)
    layout["welcome-screen"].update(Welcome())

    return layout


def trainer_ui(epoch_progress,
               batch_progress,
               url="Let W&B initialize a run!"):
    # Setup
    epoch_progress = get_epoch_table(epoch_progress)
    progress_table = get_batch_table(batch_progress)
    footer_table = get_footer_table(url)

    # Renderables
    layout = make_trainer_layout()
    layout["header"].update(Header())
    layout["epoch-level"].update(epoch_progress)
    layout["batch-level"].update(progress_table)
    layout["footer"].update(footer_table)

    return layout


if __name__ == '__main__':
    # Setup pipeline
    trainloader, testloader = setup_data_pipeline(configs)
    train_size, test_size = len(trainloader.dataset), len(testloader.dataset)
    train_num_batch, test_num_batch = len(trainloader), len(testloader)
    train_batch_size, test_batch_size = train_size//train_num_batch, test_size//test_num_batch

    # Setup progress bars
    epoch_progress, batch_progress = get_progress_bars()

    # Initialize the layout and train the model.
    layout = welcome_ui(visible=True)
    with Live(layout, refresh_per_second=1, transient=True) as live:
        # Keep the welcome page for 10 seconds.
        time.sleep(1)
        live.update(welcome_ui(visible=False))

    console = Console()

    # Ask for the dropout rate.
    prompt_ask = Text(style="white on black", overflow="fold")
    prompt_ask.append(
        "Enter the dropout rate you want the model to train with. "
        "The rate should be between "
    )
    prompt_ask.append("0 and 1 ", style="magenta")
    prompt_ask.append("with step size of 0.1")
    dropout_rate = FloatPrompt.ask(prompt=prompt_ask, 
                                    choices=[str(val) for val in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
                                    default=0.2,
                                    show_choices=False)

    # Build Model
    model = SimpleMLPModel(hidden_size=configs.model_hidden_size, dropout_rate=dropout_rate)
    # Loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Ask for W&B api token.
    wandb_ask = Confirm.ask(
        "Do you want to use your [bold blue]Weights and Biases[/bold blue] account? ")
    if wandb_ask:
        api_text = Text(style="white on black", overflow="fold")
        api_text.append("🐝 You can visit ")
        api_text.append("https://wandb.ai/authorize ", style="blue")
        api_text.append("to get your API token. Please enter your unique W&B API token: ")
        api_token = Prompt.ask(api_text, password=True)
        try:
            wandb.login(key=api_token)
        except:
            pass
        anonymous = "allow"
    else:
        anonymous = "must"

    console.rule()

    # Initialize the layout and train the model.
    layout = trainer_ui(epoch_progress, batch_progress)
    with Live(layout, refresh_per_second=1, transient=True) as live:
        # Initialize W&B run
        run = wandb.init(project='pytorch-dropout', anonymous=anonymous)

        short_url = type_tiny.tinyurl.short(run.get_url())
        live.update(trainer_ui(epoch_progress, batch_progress, short_url))

        # Train and evaluate
        epoch_progress.start_task(0)
        for epoch in range(configs.epochs):    
            train(trainloader, model, criterion, optimizer, batch_progress)
            test(testloader, model, criterion, batch_progress)
            epoch_progress.advance(0, advance=1)

        run.finish()

    if wandb_ask:
        console = Console()
        console.print("Now that the training is done, check out the [bold red]train and test[/bold red] metrics by visiting [bold blue]W&B run page[/bold blue]: ")
        console.print(short_url)
        console.print()
        console.print("How about you give it one more spin with a different dropout rate? :wink:")
        console.print()
        console.print("[orange] If you liked the work do consider giving it a heart :smile:")
    else:
        console = Console()
        console.print("Do you know how many [red] research got lost because they were not tracked properly[/red]?")
        console.print("How about we give it one more spin with your own W&B account? :wink: Visit https://wandb.ai/signup for a free account. :smile:")
        console.print()
        console.print("Since you made the effort of training the model, check out the [bold red]train and test[/bold red] metrics by visiting [bold blue]W&B run page[/bold blue]: ")
        console.print(short_url)