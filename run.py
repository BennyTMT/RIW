import os 
import json
import numpy as np 
import k_diffusion as K
import random
import sys
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont, ImageOps
from Injector import PGD_Injector , CW_Injector  , BIM_Injector
from omegaconf import OmegaConf
from stable_diffusion.ldm.util import instantiate_from_config
from argparse import ArgumentParser
import requests
import io , string 
import time 
sys.path.append("./stable_diffusion")

def load_encoder_from_config( verbose=False):
    config_path = '/u/wtd3gz/instruct-pix2pix/configs/v1-inference-attack.yaml'
    config = OmegaConf.load(config_path)
    
    ckpt = 'checkpoints/sd-v1-4.ckpt'
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
        
    return model

def load_tasks():
    f = open('four_letter.txt','r')
    waters = [w[0:4] for w in  f.readlines()]
    f.close()   
    return waters

def pixel_water(baseImg_path=None  , image = None, waters=None  , random_water = False , alpha = 0.0 ,water_idx=0  ):
    if random_water : 
        text = ''.join(random.choice(string.ascii_letters) for _ in range(3))
    else :
        text = waters[water_idx].lower()
        
    font = ImageFont.truetype('/u/wtd3gz/watermark/src/chinese.simfang.ttf', 70)
    if image == None : image = Image.open(baseImg_path)
    
    # 添加背景
    new_img = Image.new('RGBA', (image.size[0] * 3, image.size[1] * 3), (0, 0, 0, 0))
    new_img.paste(image, image.size)

    # 添加水印
    font_len = len(text)
    rgba_image = new_img.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)
        
    for i_i in range(0, rgba_image.size[0], font_len*40+75):
        for j_j in range(0, rgba_image.size[1], 140):
            i = i_i  ; j = j_j 
            #列--左右
            if i not in [470,705, 940] : continue
            #行--高度
            if j not in [ 420, 560, 700 ,840 ,980] :continue
            if j == 420 and i == 705 :  j+= 35  
            if j == 560 or j==840 : 
                if i == 470 : continue 
                if i == 705 : i -=100 
                if i == 940 : i -= 130
            if j == 840 : j += 45 
            if j == 980:  j += 25 
            if alpha < 1 : 
                image_draw.text((i, j), text, font=font, fill=(0, 0, 0, int(255*alpha)))
            else : 
                image_draw.text((i, j), text, font=font, fill=(0, 0, 0, int(alpha)))
    text_overlay = text_overlay.rotate(-45)
    image_with_text = Image.alpha_composite(rgba_image, text_overlay)
    
    # 裁切图片
    image_with_text = image_with_text.crop((image.size[0], image.size[1], image.size[0] * 2, image.size[1] * 2)).convert("RGB")
    return 2 * torch.tensor(np.array(image_with_text)).float() / 255.0 - 1.0 , text

def run_demo(injector = 'pgd'):
    # 2,3 
    # device = 'cuda:{}'.format(sys.argv[1])
    device = 'cuda:0'
    encoder_model = load_encoder_from_config()
    encoder_model.eval().to(device)
    if injector == 'pgd':
        injector = PGD_Injector(encoder_model,device=device)
    if injector == 'cw':
        injector = CW_Injector( encoder_model,device=device)
    if injector == 'bim':
        injector = BIM_Injector( encoder_model,device=device)
        
    waters = load_tasks()
    water_idx = random.randint(0, len(waters)-1)

    alpha = 0.5 
    file_path = '/u/wtd3gz/instruct-pix2pix/imgs/example.jpg'
    x_prime  , water_text  =  pixel_water(baseImg_path = file_path , waters= waters ,  random_water = False,  alpha = alpha ,  water_idx=water_idx )
    x_prime =  rearrange(x_prime, "h w c -> 1 c h w").to(encoder_model.device)
    # (0,255) --> (-1,1)
    origin_image =  2 * torch.tensor(np.array(Image.open(file_path))).float() / 255. - 1
    origin_image = rearrange(origin_image, "h w c -> 1 c h w").to(encoder_model.device)
    
    beg = time.time()
    target_image , costs  = injector.run(origin_image , x_prime= x_prime,  decoder_loss = True )
    end = time.time()
    print('it costs : ' ,  end - beg   )
    
    return costs ,  end - beg 
    # # Save target_image 
    # save_adv_path = os.path.join(DATABASE,file_name+'/{}/{}.jpg'.format(alpha_type, alpha))
    # image_save_from_tensor(target_image,save_adv_path)
    # # Save Xprime  
    # save_adv_path = os.path.join(DATABASE,file_name+'/xPrime/{}.jpg'.format(alpha))
    # image_save_from_tensor(x_prime,save_adv_path)

if __name__ == "__main__":
    for inj_type in ['bim' , 'cw' , 'pgd']:
        costs_all = []
        times=  []
        print(inj_type)
        for i in range(10):
            costs , t  = run_demo(injector = inj_type)
            costs_all.append(costs)
            times.append(t)
            np.save('inj_eff_out/{}_cost.npy'.format(inj_type) , np.array(costs_all) )
            np.save('inj_eff_out/{}_time.npy'.format(inj_type) , np.array(times) )
        print(np.mean(times))