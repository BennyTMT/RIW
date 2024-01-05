import torch , os 
import torch.nn as nn
import numpy as np 
from PIL import Image
import lpips,clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,InterpolationMode
import torch.optim as optim

class target_Injector():
    def __init__(self, model, eps=16/255, alpha=2/255, steps=1010, random_start=False, clip_loss = False,device=''):
        super(target_Injector, self).__init__()
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