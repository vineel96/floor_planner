
import pygame
from pygame.locals import *
import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_binary_mask(scene_segment, c,w,h):
    ret = np.zeros(shape=(756, 1008)) #(756, 1008) (3024, 4032)
    for i in range(h):
        for j in range(w):
            if scene_segment[i][j][0] == c:
                ret[i][j] = 1
            else:
                ret[i][j] = 0
    return ret

def get_depth_and_binary_mask_chair(scene_depth,x,y,w,h):
    dummy_scene = np.zeros(shape=(756, 1008))
    avg_depth = 0
    cell_count = 0
    for i in range(y, y + h):
        for j in range(x, x + w):
            avg_depth += scene_depth[i][j]
            dummy_scene[i][j]=1
            cell_count += 1
    avg_depth = avg_depth / cell_count
    return avg_depth, dummy_scene

if __name__=="__main__":

    w, h = 1008, 756  # original size w,h: (4032,3024)

    # getting floor coordinates
    floor_coords = set()
    scene_segment = cv2.imread("segmentation_image.png")
    scene_segment = cv2.resize(scene_segment, (w, h))
    #floor_map = np.array([[0 for i in range(1008)] for i in range(756)])  # 756x1008 (hxw)
    # for i in range(h):
    #     for j in range(w):
    #         if scene_segment[i][j][0] == 204:
    #             floor_map[i][j] = 1
    #             floor_coords.add((j, i))  # store x,y coords in the set

    # get binary maps and segment masks of each class inside scene
    binary_map = {}
    # 25:vase, 51:armchair, 76:basket, 102, 178: plaything, 127, 229, 255:wall. 153:chair, 204:floor
    classes = [25,51, 76, 102, 178, 127, 229, 255, 153, 204]
    for c in classes:
        ret_val = get_binary_mask(scene_segment,c,w,h)
        binary_map[c]=ret_val

    scene_depth = cv2.imread("depth_png.png",0)
    scene_depth = scene_depth/255
    scene_depth = cv2.resize(scene_depth,(w,h))

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    screen_rect = screen.get_rect()

    # Take chair image as input
    chair_img = pygame.image.load('chair.png')
    chair_img = pygame.transform.scale(chair_img, (170.5,333.5)) # original size (683, 1334), scaled by factor 4 (same as scene scale factor)
    chair_img.convert()
    chair_img_dup = chair_img.copy()

    chair_rect = chair_img.get_rect()
    chair_rect.center = w // 2, h-(333.5/2 + 10)

    running = True
    moving = False

    while running:
        for event in pygame.event.get():

            if event.type == QUIT:
                running = False

            elif event.type == MOUSEBUTTONDOWN:
                if chair_rect.collidepoint(event.pos): # if the mouse is pointed inside chair rect then only move
                    moving = True

            elif event.type == MOUSEBUTTONUP:
                moving = False

            # Make your image move continuously
            elif event.type == MOUSEMOTION and moving:
                x,y,w,h = chair_rect[0],chair_rect[1],chair_rect[2],chair_rect[3]
                chair_bottom_left = (chair_rect[0], chair_rect[1]+chair_rect[3])
                chair_bottom_right = (chair_rect[0] +chair_rect[2] , chair_rect[1] + chair_rect[2])
                avg_depth, chair_mask = get_depth_and_binary_mask_chair(scene_depth, x, y, w, h)

                # find if there is any intersection between chair and objects inside scene
                objects = [127]
                intersect = np.logical_and(chair_mask, binary_map[102])
                obj_intersection_depth = np.average(np.multiply(intersect,scene_depth))
                # if obj_intersection_depth > avg_depth: # lay scene first and then chair on it
                #     screen.blit(bg, (0, 0))
                #     screen.blit(chair_img, chair_rect)
                # else: # lay chair first and then scene on it
                #     screen.blit(chair_img, chair_rect)
                #     screen.blit(bg, (0, 0))

                # scale chair based on the depth
                chair_img = pygame.transform.scale(chair_img_dup, (chair_rect[2]*avg_depth, chair_rect[3]*avg_depth))
                chair_rect.move_ip(event.rel)


        bg = pygame.image.load("scene.png")
        bg = pygame.transform.scale(bg, (1008,756))
        chair_rect.clamp_ip(screen_rect)
        screen.blit(bg,(0,0))
        screen.blit(chair_img, chair_rect)
        pygame.display.update()

    pygame.quit()
