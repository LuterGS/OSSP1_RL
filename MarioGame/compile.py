from distutils.core import setup

import glob

from MarioGame.abs_filepath import ABS_PATH

setup(
    # this is the file that is run when you start the game from the command line.
    console=["main.py"],
    # data files - these are the non-python files, like images and sounds
    data_files=[
        ("sprites", glob.glob("MarioGame/sprites\\*.json")),
        ("sfx", glob.glob("MarioGame/sfx\\*.ogg") + glob.glob("MarioGame/sfx\\*.wav")),
        ("levels", glob.glob("MarioGame/levels\\*.json")),
        ("img", glob.glob("MarioGame/img\\*.gif") + glob.glob("MarioGame/img\\*.png")),
        ("", [ABS_PATH + "settings.json"]),
    ],
)
