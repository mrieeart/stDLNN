import torch
import torch.nn as nn
import numpy as np
    
'''
   UnFold patches
   x.size() =  (n, N_1, N_2, N_3, N_4)
   kernel_size = (k_0, k_1, k_2, k_3)
   y.size() = (n, L_a, L_b, k_0×k_1×k_2×k_3)
'''
def unFold(x, kernel_size, dilation=1, padding=0, stride=1): #(n, N_1, N_2, N_3, N_4)
    s  = x.size()
    x0 = x.reshape((s[0],s[1]*s[2],s[3],s[4])) #(n, N_1×N_2, N_3, N_4) 
    kb = kernel_size[2:4]
    nkb= np.prod(kb)
    x1 = nn.functional.unfold(x0, kb, dilation, padding, stride) #(n, N_1×N_2×k_2×k_3, L_b)
    nlb= x1.size()[-1]
    x1f= x1.view((s[0],s[1]*s[2],nkb,nlb)) #(n, N_1×N_2, k_2×k_3, L_b)
    x1f= x1f.permute(0,2,3,1) #(n, k_2×k_3, L_b, N_1×N_2)
    x1f= x1f.reshape((s[0],nkb*nlb,s[1],s[2])) #(n,k_2×k_3×L_b, N_1, N_2)
    ka = kernel_size[0:2]
    nka= np.prod(ka)
    x2 = nn.functional.unfold(x1f, ka, dilation, padding, stride) #(n,k_2×k_3×L_b×k_0×k_1, L_a)
    nla= x2.size()[-1]
    x2f= x2.view(s[0],nkb,nlb,nka,nla) #(n,k_2×k_3, L_b, k_0×k_1, L_a)
    x2f= x2f.permute(0,4,2,3,1)#(n, L_a, L_b, k_0×k_1, k_2×k_3)
    y  = x2f.reshape(s[0],nla,nlb,nka*nkb) #(n, L_a, L_b, k_0×k_1×k_2×k_3)
    return y


'''
   Fold patches
   y.size() = (n, L_a, L_b, k_0×k_1×k_2×k_3)
   output_size = (n, N_1, N_2, N_3, N_4)
   kernel_size = (k_0, k_1, k_2, k_3)
   x.size() =  (n, N_1, N_2, N_3, N_4)  
'''
def Fold(y, output_size, kernel_size, dilation=1, padding=0, stride=1): #(n, L_a, L_b, k_0×k_1×k_2×k_3)
    s  = output_size
    nla= y.size()[1]
    ka = kernel_size[0:2]
    nka= np.prod(ka)
    nlb= y.size()[2]
    kb = kernel_size[2:4]
    nkb= np.prod(kb)  
    x2f= y.permute(0,3,1,2) #(n, k_0×k_1×k_2×k_3, L_a, L_b)
    x2f= x2f.view((s[0],nka,nkb,nla,nlb)) #(n, k_0×k_1, k_2×k_3, L_a, L_b)
    x2 = x2f.permute((0,2,4,1,3)) #(n, k_2×k_3, L_b, k_0×k_1, L_a)
    x2 = x2.reshape((s[0],nkb*nlb*nka,nla)) #(n, k_2×k_3×L_b×k_0×k_1, L_a)
    x1f= nn.functional.fold(x2, (s[1],s[2]), ka, dilation, padding, stride) #(n, k_2×k_3×L_b, N_1, N_2)
    x1 = x1f.view((s[0],nkb,nlb,s[1]*s[2])) #(n, k_2×k_3, L_b, N_1*N_2)
    x1 = x1.permute((0,3,1,2))  #(n, N_1*N_2, k_2×k_3, L_b)
    x1 = x1.reshape((s[0],s[1]*s[2]*nkb,nlb)) #(n, N_1*N_2×k_2×k_3, L_b)
    x0 = nn.functional.fold(x1, (s[3],s[4]), kb, dilation, padding, stride) #(n, N_1×N_2, N_3, N_4)
    x  = x0.reshape((s[0],s[1],s[2],s[3],s[4]))#(n, N_1, N_2, N_3, N_4)
    return x


