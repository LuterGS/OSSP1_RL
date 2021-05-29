from distutils.core import setup

import py2exe
import glob

setup(
    # this is the file that is run when you start the game from the command line.
    console=["main.py"],
    # data files - these are the non-python files, like images and sounds
    data_files=[
        ("sprites", glob.glob("Pygame/sprites\\*.json")),
        ("sfx", glob.glob("Pygame/sfx\\*.ogg") + glob.glob("Pygame/sfx\\*.wav")),
        ("levels", glob.glob("Pygame/levels\\*.json")),
        ("img", glob.glob("Pygame/img\\*.gif") + glob.glob("Pygame/img\\*.png")),
        ("", ["settings.json"]),
    ],
)
