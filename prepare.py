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
from Injector import target_Injector 
from omegaconf import OmegaConf
from stable_diffusion.ldm.util import instantiate_from_config
from argparse import ArgumentParser
import requests
import io , string 
import time 

data_type = 'igc'

if data_type == 'igc':
    DATABASE     = 'DIR/imagic-l1/'
if data_type == 'ipx':
    DATABASE     = 'DIR/ipx-l1/'

if not os.path.exists(DATABASE): os.makedirs(DATABASE)
# 00-04 
TASKDIRS = ['DIR/shard-0{}/'.format(i) for i in range(1,4)] 

ExceptionContent =''
sys.path.append("./stable_diffusion")

def load_encoder_from_config( verbose=False):
    config_path = 'DIR/v1-inference-attack.yaml'
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
    tasks =[]
    for dir in TASKDIRS:
        for file_name in os.listdir(dir):
            tasks.append(dir + file_name)
    
    f = open('four_letter.txt','r')
    waters = [w[0:4] for w in  f.readlines()]
    f.close()   
    return tasks,waters

def down_laod_images():
    tasks,_ = load_tasks()
    count_imgs = 0 
    while True : 
        # .../shard-01/0427690
        select_file_path = tasks[random.randint(0,len(tasks))] 
        #.../shard-w/0427690'
        save_edit_path  = os.path.join(DATABASE , select_file_path.split('/')[-1])
        
        if not os.path.exists(save_edit_path): os.makedirs(save_edit_path)
        else : continue 
        json_file =  os.path.join(select_file_path ,'prompt.json')
        # Download  Images 
        with open( json_file , 'r') as file:
            data_dict = json.load(file)

            url = data_dict['url']
            try: 
                print('loading Image')
                response = requests.get(url,timeout=10)
                print('Done! ')
            except Exception as ex: 
                print('='*50)
                print('download img err : ' , ex )
                print(url)
                print('='*50)
                continue
            try:  
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                print(img.size)
                height ,withd = img.size
            except Exception as ex: 
                print('='*50)
                print('Load img err : ' , ex )
                print(url )
                print('='*50)
                continue
            if (height + withd ) /2.0 < 250.0 : continue 
            # (0,255)
            image = ImageOps.fit(img , (512, 512), method=Image.Resampling.LANCZOS)
            image.save(os.path.join(save_edit_path,'orig.jpg'))
            count_imgs+=1 
            cmd = 'cp {} {}'.format(json_file,save_edit_path)
            os.system(cmd )
            if count_imgs > 220 : exit()

def pixel_water(baseImg_path=None  , image = None, waters=None  , random_water = False , alpha = 0.0 ,water_idx=0  ):
    if random_water : 
        text = ''.join(random.choice(string.ascii_letters) for _ in range(3))
    else :
        text = waters[water_idx].lower()
        
    font = ImageFont.truetype('/u/wtd3gz/watermark/src/chinese.simfang.ttf', 70)
    if image == None : image = Image.open(baseImg_path)
    
    new_img = Image.new('RGBA', (image.size[0] * 3, image.size[1] * 3), (0, 0, 0, 0))
    new_img.paste(image, image.size)
    font_len = len(text)
    rgba_image = new_img.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)

    for i_i in range(0, rgba_image.size[0], font_len*40+75):
        for j_j in range(0, rgba_image.size[1], 140):
            i = i_i  ; j = j_j 
            if i not in [470,705, 940] : continue
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
    image_with_text = image_with_text.crop((image.size[0], image.size[1], image.size[0] * 2, image.size[1] * 2)).convert("RGB")
    return 2 * torch.tensor(np.array(image_with_text)).float() / 255.0 - 1.0 , text

def image_save_from_tensor(image, save_path=None  ):
    # (-1,1) ->  (0.,255.)
    def transfer_format(tensorImg):
        image_tensor = tensorImg.cpu().detach()
        # 2 * image / 255 - 1
        image_tensor = ( (image_tensor +1) * 125.0 ).clamp_(0, 255)
        image_np = np.squeeze(image_tensor.numpy().astype(np.uint8))
        image_np = np.transpose(image_np, (1, 2, 0))
        image_pil = Image.fromarray(image_np)
        return image_pil , image_np 
        
    image_wm , image_np = transfer_format(image)
    if save_path != None :   image_wm.save(save_path)
    return image_np
    
def record(dirName , ex ):      
    ExceptionContent =  dirName + ' Exception in open&save :%s'%ex
    with open('./expFile.txt', 'a') as f :
        f.write(ExceptionContent + '\n')
    
# prepare data from Ipx_Igc dataset 
def run_pipline_text_weight():
    # 2,3 
    device = 'cuda:{}'.format(sys.argv[1])
    encoder_model = load_encoder_from_config()
    encoder_model.eval().to(device)
    injector = target_Injector(encoder_model,device=device)
    _,waters = load_tasks()
    
    target_list = os.listdir(DATABASE)
    random.shuffle(target_list)
    best_steps =[]
    alpha_type = 'alpha'
    for file_name in target_list :
        
        if len(best_steps) != 0 : print(np.mean(best_steps)) ; print(best_steps) 
        file_path = os.path.join(DATABASE,file_name+'/orig.jpg')

        if not os.path.exists(file_path): continue
        if os.path.exists(os.path.join(DATABASE,file_name+'/{}/'.format(alpha_type))): continue 
        water_idx = random.randint(0, len(waters)-1)
        
        isFirst = True 
        for alpha in [1,5,10,15,20, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9 ,0.99]:
            print(alpha, file_name )
            # -1,1
            x_prime  , water_text  =  pixel_water(baseImg_path = file_path , waters= waters ,  random_water = False,  alpha = alpha,water_idx=water_idx )
            print(water_text)
            if isFirst : 
                with open(os.path.join(DATABASE,file_name+'/water_{}.txt'.format(alpha_type)), 'w') as f : 
                    f.write(water_text)
                isFirst = False 
                
            x_prime =  rearrange(x_prime, "h w c -> 1 c h w").to(encoder_model.device)
            # (0,255) --> (-1,1)
            origin_image =  2 * torch.tensor(np.array(Image.open(file_path))).float() / 255. - 1
            origin_image = rearrange(origin_image, "h w c -> 1 c h w").to(encoder_model.device)

            if not os.path.exists(os.path.join(DATABASE,file_name+'/{}/'.format(alpha_type))): os.makedirs(os.path.join(DATABASE,file_name+'/{}/'.format(alpha_type)))
            if not os.path.exists(os.path.join(DATABASE,file_name+'/xPrime/')): os.makedirs(os.path.join(DATABASE,file_name+'/xPrime/'))
            
            ##### Generate Target Image
            target_image , step = injector.run(origin_image , x_prime= x_prime, inject_type='pixel' , decoder_loss = False )
            best_steps.append(step)
            # Save Alpha 
            save_adv_path = os.path.join(DATABASE,file_name+'/{}/{}.jpg'.format(alpha_type, alpha))
            image_save_from_tensor(target_image,save_adv_path)
            # Save Xprime  
            save_adv_path = os.path.join(DATABASE,file_name+'/xPrime/{}.jpg'.format(alpha))
            image_save_from_tensor(x_prime,save_adv_path)

def run_pipline_encode_weight():
    # 2,3 
    device = 'cuda:{}'.format(sys.argv[1])
    encoder_model = load_encoder_from_config()
    encoder_model.eval().to(device)
    injector = target_Injector(encoder_model,device=device)
    _,waters = load_tasks()

    target_list = os.listdir(DATABASE)
    random.shuffle(target_list)
    best_steps =[]
    data_type = 'encode'
    for file_name in target_list :
        
        if len(best_steps) != 0 : print(np.mean(best_steps)) ; print(best_steps) 
        file_path = os.path.join(DATABASE,file_name+'/orig.jpg')
        
        if not os.path.exists(file_path): continue
        if os.path.exists(os.path.join(DATABASE,file_name+'/{}/'.format(data_type))): continue 
            
        water_idx = random.randint(0, len(waters)-1)
        
        isFirst = True 
        for alpha in [0.00048828125, 0.001953125, 0.00390625, 0.0078125,
                       0.03125, 0.0625, 0.125, 0.25, 0.5,  2,  8, 16, 32, 64, 128 ] :
            if not os.path.exists(os.path.join(DATABASE,file_name+'/{}/'.format(data_type))): 
                os.makedirs(os.path.join(DATABASE,file_name+'/{}/'.format(data_type)))
            print(file_name , 'decoder Loss Scale : ' , alpha,  )

            x_prime  , water_text  =  pixel_water(baseImg_path = file_path , waters= waters ,  random_water = False,  alpha = 0.5  ,water_idx=water_idx )
            x_prime =  rearrange(x_prime, "h w c -> 1 c h w").to(encoder_model.device)
            origin_image =  2 * torch.tensor(np.array(Image.open(file_path))).float() / 255. - 1
            origin_image = rearrange(origin_image, "h w c -> 1 c h w").to(encoder_model.device)

            target_image ,best_step = injector.run(origin_image , x_prime= x_prime, inject_type='pixel' , encode_loss_scale = alpha )
            best_steps.append(best_step)
            # Save Alpha 
            save_adv_path =  os.path.join(DATABASE,file_name+'/{}/{}.jpg'.format(data_type,alpha))
            image_save_from_tensor(target_image , save_path=save_adv_path)
            print(water_text)
            
            if isFirst : 
                with open(os.path.join(DATABASE,file_name+'/water_{}.txt'.format(data_type)), 'w') as f : 
                    f.write(water_text)
                isFirst = False 

# prepare data from Igc dataset 
def load_igc_tasks():
    json_dir = 'DIR/imagic-dataset/tedbench/imagic_prompts.json'
    with open(json_dir, 'r') as file : 
        tasks = json.load(file)
    return tasks
    
def build_igc_dir():
    tasks = load_igc_tasks()
    count = 0 
    save_base = 'DIR/water_edit/imagic-alphas-l1/'
    
    for one_task in tasks :
        file_path = os.path.join('DIR/clip-filtered-dataset/imagic-dataset/tedbench/originals/',one_task['fileName'])

        tar_dir = os.path.join(save_base,str(count))
        image = ImageOps.fit( Image.open(file_path).convert('RGB') , (512, 512), method=Image.Resampling.LANCZOS)
        image.save(tar_dir+'/orig.jpg')
        print(image.size)
        with open( tar_dir +'/prompt.json' , 'w') as f:
            json.dump(one_task, f, indent=4)
        print(tar_dir)

        count+=1 

if __name__ == '__main__':
    print(sys.argv[2] )
    if int(sys.argv[2]) ==7 : 
        down_laod_images()
        build_igc_dir()
    if int(sys.argv[2]) ==0 : 
        run_pipline_text_weight()
    if int(sys.argv[2]) ==1 : 
        run_pipline_encode_weight()
    