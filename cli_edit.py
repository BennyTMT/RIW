import os 
import json
import numpy as np 
import k_diffusion as K
import random
import sys
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont 
from Injector import target_Injector 
from omegaconf import OmegaConf
from stable_diffusion.ldm.util import instantiate_from_config
from argparse import ArgumentParser
import  string 
import time 


SAVEPATH = 'DIC/SAVEPATH/'
    
TASKDIRS = ['DIR/shard-0{}/'.format(i) for i in range(1,4)] 
ExceptionContent =''
sys.path.append("./stable_diffusion")

def load_encoder_from_config(verbose=False):
    config = OmegaConf.load('DIR/v1-inference-attack.yaml')
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

from myUtils import load_model_from_config , CFGDenoiser  , edit_image_with_prompt , extract_water_from_image

def load_tasks():
    tasks =[]
    for dir in TASKDIRS:
        for file_name in os.listdir(dir):
            tasks.append(dir + file_name)
    
    f = open('four_letter.txt','r')
    waters = [w[0:4] for w in  f.readlines()]
        
    return tasks,waters

def pixel_water(baseImg_path=None  , image = None, waters=None  , random_water = False  ):
    
    if random_water : 
        text = ''.join(random.choice(string.ascii_letters) for _ in range(3))
    else :
        text = waters[random.randint(0, len(waters)-1)]

    font = ImageFont.truetype('/u/wtd3gz/watermark/src/chinese.simfang.ttf', 80)
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
            image_draw.text((i, j), text, font=font, fill=(0, 0, 0, 100))
            
    '''
        # 40 100 200 
        for i in range(0, rgba_image.size[0], font_len*40+60):
            for j in range(0, rgba_image.size[1], 160):
                # image_draw.text((i, j), text, font=font, fill='white')
                image_draw.text((i, j), text, font=font, fill=(0, 0, 0, 100))
    '''
    
    text_overlay = text_overlay.rotate(-45)
    image_with_text = Image.alpha_composite(rgba_image, text_overlay)

    # 裁切图片
    image_with_text = image_with_text.crop((image.size[0], image.size[1], image.size[0] * 2, image.size[1] * 2)).convert("RGB")
    return 2 * torch.tensor(np.array(image_with_text)).float() / 255.0 - 1.0 , text

def image_save_from_tensor(image, save_path ):
    def transfer_format(tensorImg):
        image_tensor = tensorImg.cpu().detach()
        # 2 * image / 255 - 1
        image_tensor = ( (image_tensor +1) * 125.0 ).clamp_(0, 255)
        image_np = np.squeeze(image_tensor.numpy().astype(np.uint8))
        image_np = np.transpose(image_np, (1, 2, 0))
        image_pil = Image.fromarray(image_np)
        return image_pil , image_np 
        
    image_wm , image_np = transfer_format(image)
    image_wm.save(save_path)
    return image_np

def record(dirName , ex ):      
    ExceptionContent =  dirName + ' Exception in open&save :%s'%ex
    with open('./expFile.txt', 'a') as f :
        f.write(ExceptionContent + '\n')

def load_editModel_args():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=False, type=str)
    parser.add_argument("--output", required=False, type=str)
    parser.add_argument("--edit" , default="turn him into a cyborg", required=False, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda", type=int)
    args = parser.parse_args()
    
    torch.cuda.set_device(args.cuda)
    
    args.edit = "turn him into a cyborg"
    
    config = OmegaConf.load(args.config)
    editModel = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    editModel.eval().cuda()
    
    model_wrap = K.external.CompVisDenoiser(editModel)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = editModel.get_learned_conditioning([""])
    device = next(editModel.parameters()).device
    
    return device , editModel , null_token, args , model_wrap_cfg ,model_wrap
    
def run_pipline():
    device , edit_model , null_token, args , model_wrap_cfg ,model_wrap  = load_editModel_args()
    
    encoder_model = load_encoder_from_config()
    encoder_model.eval().to(device)
    seed = 100 
    while True :
        target_list = os.listdir(SAVEPATH)
        random.shuffle(target_list)
        isThereTask = False 
        data_type = 'encode'
        for file_name in  target_list :
            save_edit_path = os.path.join(SAVEPATH,file_name)
            print(save_edit_path)
            if not os.path.exists(os.path.join(SAVEPATH,file_name+'/{}/'.format(data_type))): 
                continue 
            if not os.path.exists(os.path.join(SAVEPATH,file_name+'/water_{}.txt'.format(data_type))) :
                continue
            if not os.path.exists(os.path.join(save_edit_path, 'ipx_{}'.format(data_type))): os.makedirs(os.path.join(save_edit_path, 'ipx_{}'.format(data_type)))
            else :   continue 
                
            json_file =  os.path.join(save_edit_path,'prompt.json')
            with open( json_file , 'r') as file:
                data_dict = json.load(file)
                args.edit = data_dict['edit']
                
            isThereTask = True
            for alpha in  [1,  2,  8, 16, 32, 64, 128 ] :
                try : 
                    target_image = Image.open(os.path.join(SAVEPATH,file_name+'/{}/{}.jpg'.format(data_type,alpha)))
                except:
                    continue
                target_image = (2.0*  (torch.tensor(np.array(target_image)).float()) /255.0 ) -1 
                target_image = rearrange(target_image, "h w c -> 1 c h w").to(encoder_model.device)
                edit_image_with_prompt(edit_model,args,target_image,null_token ,model_wrap,model_wrap_cfg,seed, os.path.join(save_edit_path, 'ipx_{}/{}.jpg'.format(data_type,alpha)))

        if not isThereTask : 
            print('There is not task, sleeping 3 mins...')
            time.sleep(60*3)
            
if __name__ == '__main__':
    run_pipline()
