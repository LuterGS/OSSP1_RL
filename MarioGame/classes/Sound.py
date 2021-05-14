from pygame import mixer
from MarioGame.abs_filepath import ABS_PATH


class Sound:
    def __init__(self):
        self.music_channel = mixer.Channel(0)
        self.music_channel.set_volume(0.2)
        self.sfx_channel = mixer.Channel(1)
        self.sfx_channel.set_volume(0.2)

        self.allowSFX = True

        self.soundtrack = mixer.Sound(ABS_PATH + "sfx/main_theme.ogg")
        self.coin = mixer.Sound(ABS_PATH + "sfx/coin.ogg")
        self.bump = mixer.Sound(ABS_PATH + "sfx/bump.ogg")
        self.stomp = mixer.Sound(ABS_PATH + "sfx/stomp.ogg")
        self.jump = mixer.Sound(ABS_PATH + "sfx/small_jump.ogg")
        self.death = mixer.Sound(ABS_PATH + "sfx/death.wav")
        self.kick = mixer.Sound(ABS_PATH + "sfx/kick.ogg")
        self.brick_bump = mixer.Sound(ABS_PATH + "sfx/brick-bump.ogg")
        self.powerup = mixer.Sound(ABS_PATH + 'sfx/powerup.ogg')
        self.powerup_appear = mixer.Sound(ABS_PATH + 'sfx/powerup_appears.ogg')
        self.pipe = mixer.Sound(ABS_PATH + 'sfx/pipe.ogg')

    def play_sfx(self, sfx):
        if self.allowSFX:
            self.sfx_channel.play(sfx)

    def play_music(self, music):
        self.music_channel.play(music)
