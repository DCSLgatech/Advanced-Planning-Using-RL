# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:38:26 2017

@author: CYOU7
"""

import pygame
pygame.init()
def make_video(screen):
    _image_num = 0
    while True:
        _image_num += 1
        str_num = "0000" + str(_image_num)
        file_name = "image" + str_num[-5:] + ".jpg"
        pygame.image.save(screen, "ScreenFolder/"+file_name)
#        print("In generator ", file_name)  # delete, just for demonstration
#        pygame.time.wait(1000)  # delete, just for demonstration
        yield
