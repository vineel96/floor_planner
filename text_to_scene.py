import torch
import cv2
import numpy as np
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionDepth2ImgPipeline

if __name__=="__main__":
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    w, h = 1008, 756
    scene = cv2.imread("scene.png")
    scene = cv2.resize(scene, (w, h))

    scene_depth = cv2.imread("depth_png.png")
    scene_depth = cv2.resize(scene_depth, (w, h))
    #scene_depth = torch.from_numpy(scene_depth).float() / 255.0
    #scene_depth = scene_depth.permute(2, 0, 1)
    scene_depth = np.expand_dims(scene_depth, axis=0)

    generated_image = pipe("room with three bean chairs and table", image = scene_depth).images[0]
    cv2.imwrite("Generated_image.png", np.array(generated_image))
    #cv2.imshow("Generated image", np.array(generated_image))
    #cv2.waitKey(0)