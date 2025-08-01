from scludam.cli_analysis import main as analysis_menu
from scludam.cli_input import select_input
from scludam.cli_utils import (
    prompt_cli_selector,
)

di = None


def launch():
    """Launch the SCLUDAM CLI.

    This function serves as the entry point for the command-line interface of SCLUDAM.
    """    
    global di
    print("Welcome to SCLUDAM CLI")
    print("----------------------")
    if di is None:
        print("No dataset loaded")
        option_names = ["Load New Dataset", "Exit"]
        options = ["load", "exit"]
    else:
        print(di)
        option_names = [
            "Load New Dataset",
            "Membership Analysis",
            "Clear Console",
            "Exit",
        ]
        options = ["load", "analysis", "clear", "exit"]
    print("----------------------")

    selected = prompt_cli_selector(
        "Options:",
        option_names,
        options,
    )
    if selected == "exit":
        return
    if selected == "clear":
        from os import name, system

        def clear():
            # for windows
            if name == "nt":
                _ = system("cls")

            # for mac and linux(here, os.name is 'posix')
            else:
                _ = system("clear")

        clear()
        launch()
    if selected == "load":
        di = select_input()
        launch()
    if selected == "analysis":
        analysis_menu(di)
        launch()
