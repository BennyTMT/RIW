import torch , os 
import torch.nn as nn
import numpy as np 
from PIL import Image
import lpips,clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,InterpolationMode
import torch.optim as optim

class random_Injector():
    def __init__(self, model, eps=8/255, alpha=2/255, steps=500, random_start=False, save_name ='text'):
        super(random_Injector, self).__init__()
        self.model = model 
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.save_name = save_name 
        self.auto_bound= False
        
        self.l2_loss = nn.MSELoss()
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.clip_model , _ =  clip.load("ViT-L/14", device="cuda")
        # 224 
        n_px= self.clip_model.visual.input_resolution
        self.transform_ = Compose([
                Resize(n_px , interpolation=InterpolationMode.BICUBIC),
                CenterCrop(n_px),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        
    def run(self, origin_imgs , std_dev = 0.1  ):
        
        self.std_dev = std_dev
        org_images = origin_imgs.clone().detach().to('cuda') 
        
        if self.random_start:
            # Starting at a uniformly random point
            iwm_images = iwm_images + \
                torch.empty_like(iwm_images).uniform_(-self.eps, self.eps)
            iwm_images = torch.clamp(iwm_images, min=-1, max=1).detach()
        else:    
            iwm_images = origin_imgs.clone().detach().to('cuda') 
             
        if self.auto_bound:
            low_bound = float(torch.min( origin_imgs.clone().detach().to('cuda') ))
            upp_bound = float(torch.max( origin_imgs.clone().detach().to('cuda') ))
        else:
            low_bound , upp_bound = -1. , 1.  

        for i in range(0, self.steps +1):
            iwm_images.requires_grad = True
            
            imw_image_features  = self.model.encode_first_stage(iwm_images).mode()
            org_image_features  = self.model.encode_first_stage(org_images).mode()
            
            cost = -self.loss_fn( origin_imgs.clone().detach()  ,  iwm_images , org_image_features , imw_image_features)
            grad = torch.autograd.grad(cost, iwm_images,
                                        retain_graph=False, create_graph=False)[0]

            iwm_images = iwm_images.detach() + self.alpha*grad.sign()
            delta      = torch.clamp(iwm_images - origin_imgs , min=-self.eps, max= self.eps)
            iwm_images = torch.clamp(origin_imgs + delta, min=low_bound, max=upp_bound).detach()
            
            if i%20 == 0 : print(i , cost  )
            # if i%100 == 0 : self.image_save(iwm_images.clone() , origin_imgs.clone() , str(i) , need_denormalize= False ) 
            if i==250 : return  iwm_images.clone()
                
    def check_loaded_encoder(self,  adv_image_features, tar_image_features , store =  True ):
        print(adv_image_features.shape)
        print(tar_image_features.shape)
        tar_ecodering = tar_image_features.data.cpu().numpy()

        if store : np.save('tar_encode.npy' , tar_ecodering  ) ; print('saved!') ; exit()
        else:      
            tar_ecodering_saved = np.load('tar_encode.npy' )
            a = tar_ecodering - tar_ecodering_saved
            print(np.mean( np.abs(a)))
            print(np.mean( np.abs(tar_ecodering_saved)))
            print(a.shape)
            exit()
            
    def loss_fn(self, orig_image  ,  iwm_image , orig_encode , iwm_encode):
        
        c1 = 1 
        c2 = 3
        c3 = 50 
        
        l1 = torch.abs(self.lpips(orig_image,iwm_image))
        l2 = self.clip_distance(  orig_image,iwm_image)
        l3 = self.l2_loss(orig_encode + torch.randn_like(orig_encode) * self.std_dev , iwm_encode  )
        # print(l1 , l2 , l3 )
        return c1* l1 + c2 * l2  + c3 * l3 
        
    def clip_distance(self , x_0 , x  ):
        x_0 = self.transform_(x_0) ; x = self.transform_(x)
        orig_image_features  = self.clip_model.encode_image(x_0)
        targ_image_features  = self.clip_model.encode_image(x)
        return  self.l2_loss(orig_image_features ,targ_image_features )

    def image_save(self, image_tensor, origin_tensor , name=-1 , need_denormalize = False):
        IMGPATH = '/YOURDIR/instruct-pix2pix/imgs/wm/'+ str(self.std_dev) +'/'
        
        if not os.path.exists(IMGPATH ): os.makedirs(IMGPATH )
        
        def transfer_format(tensorImg):
            image_tensor = tensorImg.cpu().detach()
            # 2 * image / 255 - 1
            image_tensor = ( (image_tensor +1) * 125.0 ).clamp_(0, 255)
            image_np = np.squeeze(image_tensor.numpy().astype(np.uint8))
            image_np = np.transpose(image_np, (1, 2, 0))
            image_pil = Image.fromarray(image_np)
            return image_pil , image_np 
            
        image_wm , image_np = transfer_format(image_tensor)
        image_wm.save(IMGPATH+str(name)+"_tar_img.png")
        
        orig_img , orig_np = transfer_format(origin_tensor)
        orig_img.save(IMGPATH+str(name)+"_orig_img.png")
        
        image_water = Image.fromarray(image_np - orig_np )
        print(image_np - orig_np )
        image_water.save(IMGPATH+str(name)+"_water_img.png")

class PGD_Injector():
    def __init__(self, model, eps=16/255, alpha=2/255, steps=1010, random_start=False, clip_loss = False,device=''):
        super(PGD_Injector, self).__init__()
        self.clip_loss = False
        self.model = model
        self.device= device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.auto_bound= True
        self.l2_loss = nn.MSELoss()
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)
        if clip_loss: 
            self.clip_loss = clip_loss
            self.clip_model , _ =  clip.load("ViT-L/14", device=self.device)
            # 224 
            n_px= self.clip_model.visual.input_resolution
            self.transform_ = Compose([
                    Resize(n_px , interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(n_px),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])

    def run(self, origin_imgs, x_prime = None , water= None  ,  
            noised_degree_latent=0 ,  decoder_loss = False , lambda3 =1 ,  lambda2 = 1.0 ):

        self.decoder_loss = decoder_loss 
        self.origin_imgs  = origin_imgs 
        self.lambda3  = lambda3 
        self.lambda2  = lambda2 
        
        self.decoder_x_prime = None 
        self.noised_degree = noised_degree_latent

        print('lambda : '  , self.lambda2 , self.lambda3 )
        
        if x_prime != None :  
            tar_images = x_prime.clone().detach().to(self.device) 
        if water!= None :     
            e_water = self.model.encode_first_stage(water.clone().detach().to(self.device) ).mode()
            
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=-1, max=1).detach()
        else:    
            adv_images = origin_imgs.clone().detach().to(self.model.device) 

        if self.auto_bound:
            low_bound = float(torch.min( origin_imgs.clone().detach().to(self.device) ))
            upp_bound = float(torch.max( origin_imgs.clone().detach().to(self.device) ))
        else:
            low_bound , upp_bound = -1. , 1.  

        lowest_cost = 1000 
        return_image = None 
        self.tar_image_features = self.model.encode_first_stage(tar_images).mode()
        costs = []
        for i in range(1, self.steps +1):
            adv_images.requires_grad = True
            adv_image_features = self.model.encode_first_stage(adv_images).mode()
            
            cost = -self.loss_fn(adv_images ,origin_imgs  , adv_image_features, self.tar_image_features)

            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta      = torch.clamp(adv_images - origin_imgs , min=-self.eps, max= self.eps)
            adv_images = torch.clamp(origin_imgs + delta, min=low_bound, max=upp_bound).detach()
            
            if i%50 == 0 : 
                curr_cost = abs(float( cost.detach().data.cpu().numpy() ) )
                print(i , curr_cost ) 
                costs.append(curr_cost)
                # self.image_save(adv_images.clone(), i , inject_type , noised_degree = noised_degree ) 
                # if lowest_cost > curr_cost : 
                #     return_image = adv_images.clone().detach()
                #     lowest_cost  = curr_cost
                #     best_step = i 
        # return return_image, best_step 
        return return_image, costs      

    def check_loaded_encoder(self,  adv_image_features, tar_image_features , store =  True ):
        print(adv_image_features.shape)
        print(tar_image_features.shape)
        tar_ecodering = tar_image_features.data.cpu().numpy()
        
        if store : np.save('tar_encode.npy' , tar_ecodering  ) ; print('saved!') ; exit()
        else:      
            tar_ecodering_saved = np.load('tar_encode.npy' )
            a = tar_ecodering - tar_ecodering_saved
            print(np.mean( np.abs(a)))
            print(np.mean( np.abs(tar_ecodering_saved)))
            print(a.shape)
            exit()
    
    def loss_fn(self, adv_image  ,  ori_image , adv_encode , tar_encode ):
        c1 = 1 
        c2 = self.lambda2
        c3 = self.lambda3 

        l1 = torch.abs(self.lpips(adv_image,ori_image) )
        l2 = torch.mean(torch.abs(adv_encode - tar_encode ))
        if self.decoder_loss  :
            adv_encode = (adv_encode-adv_encode.mean())/adv_encode.std()
            loss_decoder = self.l2_loss( self.origin_imgs,  self.model.decode_first_stage(adv_encode))
        # if self.clip_loss :    
        #     l3 = self.clip_distance(  ori_image ,adv_image )
        #     return  c1* l1 + c2 * l2 + c3* l3
        if self.decoder_loss :
            return  c1* l1 + c2 * l2 + c3 * loss_decoder 
        else : 
            print('there will be no punishment!')
            return  c1* l1 + c2 * l2 
        
    def clip_distance(self , x_0 , x  ):
        x_0 = self.transform_(x_0) ; x = self.transform_(x)
        orig_image_features  = self.clip_model.encode_image(x_0)
        targ_image_features  = self.clip_model.encode_image(x)
        return  self.l2_loss(orig_image_features ,targ_image_features )

    def image_save(self, image_tensor , step , inject_type , noised_degree = 0 ): 
        image_tensor = image_tensor.cpu().detach()

        IMGPATH = '/YOURDIR/instruct-pix2pix/imgs/injection/{}/'.format(inject_type)
        if not os.path.exists(IMGPATH): os.makedirs(IMGPATH)
        
        # image_tensor = (0,1)
        image_tensor = ( (image_tensor +1) * 125.0 ).clamp_(0, 255)
        image_np = np.squeeze(image_tensor.numpy().astype(np.uint8))
        # Transpose the NumPy array from (C, H, W) to (H, W, C)
        image_np = np.transpose(image_np, (1, 2, 0))
        # Convert the NumPy array to a PIL Image object
        image_pil = Image.fromarray(image_np)
        # Save the PIL Image object
        image_pil.save(IMGPATH+"{}_{}.png".format(noised_degree,step))
        
    def image_save_from_tensor(self, tensorImg, save_path ):
        image_tensor = tensorImg.cpu().detach()
        # 2 * image / 255 - 1
        image_tensor = ( (image_tensor +1) * 125.0 ).clamp_(0, 255)
        image_np = np.squeeze(image_tensor.numpy().astype(np.uint8))
        image_np = np.transpose(image_np, (1, 2, 0))
        image_pil = Image.fromarray(image_np)
        image_pil.save(save_path)

class CW_Injector():
    def __init__(self, model, c=1, steps=1010, lr=0.01 ,  clip_loss = False,device='' ):
        super(CW_Injector, self).__init__()
        self.c = c
        self.steps = steps
        self.lr = lr

        self.clip_loss = False
        self.model = model
        self.device= device
        self.steps = steps
        self.random_start = False
        self.auto_bound= True
        self.l2_loss = nn.MSELoss()
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)
        if clip_loss: 
            self.clip_loss = clip_loss
            self.clip_model , _ =  clip.load("ViT-L/14", device=self.device)
            # 224 
            n_px= self.clip_model.visual.input_resolution
            self.transform_ = Compose([
                    Resize(n_px , interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(n_px),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
    def run(self, origin_imgs, x_prime = None , 
            noised_degree_latent=0 ,  decoder_loss = False , lambda3 =1 ,  lambda2 = 1.0 ):
        
        self.decoder_loss = decoder_loss 
        self.origin_imgs  = origin_imgs 
        self.lambda3  = lambda3 
        self.lambda2  = lambda2 

        self.decoder_x_prime = None 
        self.noised_degree = noised_degree_latent

        if x_prime != None :    tar_images = x_prime.clone().detach().to(self.device) 
        self.tar_image_features = self.model.encode_first_stage(tar_images).mode()
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(origin_imgs).detach()
        w.requires_grad = True
        
        best_adv_images = origin_imgs.clone().detach().to(self.model.device) 

        optimizer = optim.Adam([w], lr=self.lr)

        costs = []
        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            adv_image_features = self.model.encode_first_stage(adv_images).mode()
            cost = self.loss_fn(adv_images ,origin_imgs  , adv_image_features, self.tar_image_features)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            cost_v = float(cost.detach().data.cpu().numpy()) 
            if step % 10 == 0 : 
                costs.append(cost_v)
            if step % 50 == 0 :    
                print(step , cost_v)
        return best_adv_images,  costs 
    def loss_fn(self, adv_image  ,  ori_image , adv_encode , tar_encode ):
        c1 = 1 
        c2 = self.lambda2
        c3 = self.lambda3 

        l1 = torch.abs(self.lpips(adv_image,ori_image) )
        l2 = torch.mean(torch.abs(adv_encode - tar_encode ))
        if self.decoder_loss  :
            adv_encode = (adv_encode-adv_encode.mean())/adv_encode.std()
            loss_decoder = self.l2_loss( self.origin_imgs,  self.model.decode_first_stage(adv_encode))
        # if self.clip_loss :    
        #     l3 = self.clip_distance(  ori_image ,adv_image )
        #     return  c1* l1 + c2 * l2 + c3* l3
        if self.decoder_loss :
            return  c1* l1 + c2 * l2 + c3 * loss_decoder 
        else : 
            print('there will be no punishment!')
            return  c1* l1 + c2 * l2 

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

class BIM_Injector():
    def __init__(self, model, alpha=2/255, eps=16/255, steps=1010, random_start=False, clip_loss = False,device=''):
        super(BIM_Injector, self).__init__()
        self.clip_loss = False
        self.model = model
        self.device= device
        self.alpha = alpha
        self.eps = eps
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.auto_bound= True
        self.l2_loss = nn.MSELoss()
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)
        if clip_loss: 
            self.clip_loss = clip_loss
            self.clip_model , _ =  clip.load("ViT-L/14", device=self.device)
            # 224 
            n_px= self.clip_model.visual.input_resolution
            self.transform_ = Compose([
                    Resize(n_px , interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(n_px),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])

    def run(self, origin_imgs, x_prime = None , water= None  ,  
            noised_degree_latent=0 ,  decoder_loss = False , lambda3 =1 ,  lambda2 = 1.0 ):

        self.decoder_loss = decoder_loss 
        self.origin_imgs  = origin_imgs 
        self.lambda3  = lambda3 
        self.lambda2  = lambda2 
        
        self.decoder_x_prime = None 
        self.noised_degree = noised_degree_latent

        print('lambda : '  , self.lambda2 , self.lambda3 )
        
        if x_prime != None :  
            tar_images = x_prime.clone().detach().to(self.device) 
        if water!= None :     
            e_water = self.model.encode_first_stage(water.clone().detach().to(self.device) ).mode()
            
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=-1, max=1).detach()
        else:    
            adv_images = origin_imgs.clone().detach().to(self.model.device) 
            
        if self.auto_bound:
            low_bound = float(torch.min( origin_imgs.clone().detach().to(self.device) ))
            upp_bound = float(torch.max( origin_imgs.clone().detach().to(self.device) ))
        else:
            low_bound , upp_bound = -1. , 1.  

        return_image = None 
        self.tar_image_features = self.model.encode_first_stage(tar_images).mode()
        costs = []
        for i in range(1, self.steps +1):
            adv_images.requires_grad = True
            adv_image_features = self.model.encode_first_stage(adv_images).mode()

            cost = -self.loss_fn(adv_images ,origin_imgs  , adv_image_features, self.tar_image_features)

            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            
            a      = torch.clamp(adv_images - origin_imgs , min=0)
            b = (adv_images >= a).float() * adv_images + (
                adv_images < a
            ).float() * a  # nopep8
            c = (b > origin_imgs + self.eps).float() * (origin_imgs + self.eps) + (
                b <= origin_imgs + self.eps
            ).float() * b  # nopep8
            adv_images = torch.clamp(c, max=1).detach()
            
            if i%50 == 0 : 
                curr_cost = abs(float( cost.detach().data.cpu().numpy() ) )
                print(i , curr_cost ) 
                costs.append(curr_cost)
        return adv_images, costs      

    def check_loaded_encoder(self,  adv_image_features, tar_image_features , store =  True ):
        print(adv_image_features.shape)
        print(tar_image_features.shape)
        tar_ecodering = tar_image_features.data.cpu().numpy()
        
        if store : np.save('tar_encode.npy' , tar_ecodering  ) ; print('saved!') ; exit()
        else:      
            tar_ecodering_saved = np.load('tar_encode.npy' )
            a = tar_ecodering - tar_ecodering_saved
            print(np.mean( np.abs(a)))
            print(np.mean( np.abs(tar_ecodering_saved)))
            print(a.shape)
            exit()
    
    def loss_fn(self, adv_image  ,  ori_image , adv_encode , tar_encode ):
        c1 = 1 
        c2 = self.lambda2
        c3 = self.lambda3 

        l1 = torch.abs(self.lpips(adv_image,ori_image) )
        l2 = torch.mean(torch.abs(adv_encode - tar_encode ))
        if self.decoder_loss  :
            adv_encode = (adv_encode-adv_encode.mean())/adv_encode.std()
            loss_decoder = self.l2_loss( self.origin_imgs,  self.model.decode_first_stage(adv_encode))
        # if self.clip_loss :    
        #     l3 = self.clip_distance(  ori_image ,adv_image )
        #     return  c1* l1 + c2 * l2 + c3* l3
        if self.decoder_loss :
            return  c1* l1 + c2 * l2 + c3 * loss_decoder 
        else : 
            print('there will be no punishment!')
            return  c1* l1 + c2 * l2 
        
    def clip_distance(self , x_0 , x  ):
        x_0 = self.transform_(x_0) ; x = self.transform_(x)
        orig_image_features  = self.clip_model.encode_image(x_0)
        targ_image_features  = self.clip_model.encode_image(x)
        return  self.l2_loss(orig_image_features ,targ_image_features )

    def image_save(self, image_tensor , step , inject_type , noised_degree = 0 ): 
        image_tensor = image_tensor.cpu().detach()

        IMGPATH = '/YOURDIR/instruct-pix2pix/imgs/injection/{}/'.format(inject_type)
        if not os.path.exists(IMGPATH): os.makedirs(IMGPATH)
        
        # image_tensor = (0,1)
        image_tensor = ( (image_tensor +1) * 125.0 ).clamp_(0, 255)
        image_np = np.squeeze(image_tensor.numpy().astype(np.uint8))
        # Transpose the NumPy array from (C, H, W) to (H, W, C)
        image_np = np.transpose(image_np, (1, 2, 0))
        # Convert the NumPy array to a PIL Image object
        image_pil = Image.fromarray(image_np)
        # Save the PIL Image object
        image_pil.save(IMGPATH+"{}_{}.png".format(noised_degree,step))
        
    def image_save_from_tensor(self, tensorImg, save_path ):
        image_tensor = tensorImg.cpu().detach()
        # 2 * image / 255 - 1
        image_tensor = ( (image_tensor +1) * 125.0 ).clamp_(0, 255)
        image_np = np.squeeze(image_tensor.numpy().astype(np.uint8))
        image_np = np.transpose(image_np, (1, 2, 0))
        image_pil = Image.fromarray(image_np)
        image_pil.save(save_path)
        
        