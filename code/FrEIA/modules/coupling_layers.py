from math import exp

import torch
import torch.nn as nn

from .coeff_functs import F_conv, F_fully_connected, F_GRU, F_GRU_BN, F_cgate


class rev_layer(nn.Module):
    '''General reversible layer modeled after the lifting scheme. Uses some
    non-reversible transformation F, but splits the channels up to make it
    revesible (see lifting scheme). F itself does not have to be revesible. See
    F_* classes above for examples.'''

    def __init__(self, dims_in, F_class=F_conv, F_args={}):
        super(rev_layer, self).__init__()
        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.F = F_class(self.split_len2, self.split_len1, **F_args)
        self.G = F_class(self.split_len1, self.split_len2, **F_args)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2 + self.G(y1)
        else:
            y2 = x2 - self.G(x1)
            y1 = x1 - self.F(y2)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        return torch.zeros(x.shape[0])

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class rev_multiplicative_layer(nn.Module):
    '''The RevNet block is not a general function approximator. The reversible
    layer with a multiplicative term presented in the real-NVP paper is much
    more general. This class uses some non-reversible transformation F, but
    splits the channels up to make it revesible (see lifting scheme). F itself
    does not have to be revesible. See F_* classes above for examples.'''

    def __init__(self, dims_in, F_class=F_fully_connected, F_args={},
                 clamp=5.):
        super(rev_multiplicative_layer, self).__init__()
        channels = dims_in[0][0]

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.ndims = len(dims_in[0])

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2, **F_args)
        self.t1 = F_class(self.split_len1, self.split_len2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1, **F_args)
        self.t2 = F_class(self.split_len2, self.split_len1, **F_args)

    def e(self, s):
        # return torch.exp(torch.clamp(s, -self.clamp, self.clamp))
        # return (self.max_s-self.min_s) * torch.sigmoid(s) + self.min_s
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = self.e(self.s2(x2)) * x1 + self.t2(x2)
            y2 = self.e(self.s1(y1)) * x2 + self.t1(y1)
        else:  # names of x and y are swapped!
            y2 = (x2 - self.t1(x1)) / self.e(self.s1(x1))
            y1 = (x1 - self.t2(y2)) / self.e(self.s2(y2))
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            s2 = self.s2(x2)
            y1 = self.e(s2) * x1 + self.t2(x2)
            jac = self.log_e(self.s1(y1)) + self.log_e(s2)
        else:
            s1 = self.s1(x1)
            y2 = (x2 - self.t1(x1)) / self.e(s1)
            jac = -self.log_e(s1) - self.log_e(self.s2(y2))

        return torch.sum(jac, dim=tuple(range(1, self.ndims+1)))

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

# class rev_recurrent_layer(nn.Module):

#     def __init__(self, dims_in, F_class=F_GRU, F_args={},
#                  clamp=5.):
#         super(rev_recurrent_layer, self).__init__()
#         channels = dims_in[0][0]

#         self.split_len1 = channels // 2
#         self.split_len2 = channels - channels // 2
#         self.ndims = len(dims_in[0])

#         self.clamp = clamp
#         self.max_s = exp(clamp)
#         self.min_s = exp(-clamp)

#         self.s1 = F_class(self.split_len1, self.split_len2, **F_args)
#         self.t1 = F_class(self.split_len1, self.split_len2, **F_args)
#         self.s2 = F_class(self.split_len2, self.split_len1, **F_args)
#         self.t2 = F_class(self.split_len2, self.split_len1, **F_args)

#     def e(self, s):
#         # return torch.exp(torch.clamp(s, -self.clamp, self.clamp))
#         # return (self.max_s-self.min_s) * torch.sigmoid(s) + self.min_s
#         return torch.exp(self.clamp * 0.636 * torch.atan(s))

#     def log_e(self, s):
#         '''log of the nonlinear function e'''
#         return self.clamp * 0.636 * torch.atan(s)

#     def forward(self, x, h_s1, h_t1, h_s2, h_t2, rev=False):
#         x1, x2 = (x[0].narrow(1, 0, self.split_len1),
#                   x[0].narrow(1, self.split_len1, self.split_len2))

#         if not rev:
#             s2_x2, h_s2_next = self.s2(x2, h_s2)
#             t2_x2, h_t2_next = self.t2(x2, h_t2)
#             y1 = self.e(s2_x2) * x1 + t2_x2
            
#             s1_y1, h_s1_next = self.s1(y1, h_s1)
#             t1_y1, h_t1_next = self.t1(y1, h_t1)
#             y2 = self.e(s1_y1) * x2 + t1_y1
#         else:  # names of x and y are swapped!
            
#             s1_x1, h_s1_next = self.s1(x1, h_s1)
#             t1_x1, h_t1_next = self.t1(x1, h_t1)
#             y2 = (x2 - t1_x1) / self.e(s1_x1)
            
#             s2_y2, h_s2_next = self.s2(y2, h_s2)
#             t2_y2, h_t2_next = self.t2(y2, h_t2)
#             y1 = (x1 - t2_y2) / self.e(s2_y2)
#         return [torch.cat((y1, y2), 1)], h_s1_next, h_t1_next, h_s2_next, h_t2_next

#     def jacobian(self, x, h_s1, h_t1, h_s2, h_t2, rev=False):
#         x1, x2 = (x[0].narrow(1, 0, self.split_len1),
#                   x[0].narrow(1, self.split_len1, self.split_len2))
        
#         if not rev:
#             s2_x2, h_s2_next = self.s2(x2, h_s2)
#             t2_x2, h_t2_next = self.t2(x2, h_t2)
#             y1 = self.e(s2_x2) * x1 + t2_x2
            
#             s1_y1, h_s1_next = self.s1(y1, h_s1)
#             t1_y1, h_t1_next = self.t1(y1, h_t1)
            
#             jac = self.log_e(s1_y1) + self.log_e(s2_x2)
#         else:
#             s1_x1, h_s1_next = self.s1(x1, h_s1)
#             t1_x1, h_t1_next = self.t1(x1, h_t1)
#             y2 = (x2 - t1_x1) / self.e(s1_x1)
            
#             s2_y2, h_s2_next = self.s2(y2, h_s2)
#             t2_y2, h_t2_next = self.t2(y2, h_t2)
            
#             jac = -self.log_e(s1_x1) - self.log_e(s2_y2)

#         return torch.sum(jac, dim=tuple(range(1, self.ndims+1))), h_s1_next, h_t1_next, h_s2_next, h_t2_next 

#     def output_dims(self, input_dims):
#         assert len(input_dims) == 1, "Can only use 1 input"
#         return input_dims

class rev_gru_layer(nn.Module):

    def __init__(self, dims_in, F_class=F_GRU, F_args={},
                 clamp=5.):
        super(rev_gru_layer, self).__init__()
        channels = dims_in[0][0]

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.ndims = len(dims_in[0])

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2, **F_args)
        self.t1 = F_class(self.split_len1, self.split_len2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1, **F_args)
        self.t2 = F_class(self.split_len2, self.split_len1, **F_args)

    def e(self, s):
        # return torch.exp(torch.clamp(s, -self.clamp, self.clamp))
        # return (self.max_s-self.min_s) * torch.sigmoid(s) + self.min_s
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    # Single recurrent step through t. Explicitly output hidden states for t+1.
    def recurrent_block(self, x, h_s1, h_t1, h_s2, h_t2, rev=False):
        # print("x.shape in recurrent_block: ", x.shape)
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
#         print("x1.shape in recurrent_block: ", x1.shape)
#         print("x2.shape in recurrent_block: ", x2.shape)
#         print("h_s1.shape: ", h_s1.shape)

        if not rev:
            # print(x2.is_cuda, h_s2.is_cuda)
            s2_x2, h_s2_next = self.s2(x2, h_s2)
            t2_x2, h_t2_next = self.t2(x2, h_t2)
            y1 = self.e(s2_x2) * x1 + t2_x2
            
            s1_y1, h_s1_next = self.s1(y1, h_s1)
            t1_y1, h_t1_next = self.t1(y1, h_t1)
            y2 = self.e(s1_y1) * x2 + t1_y1
        else:  # names of x and y are swapped!
            s1_x1, h_s1_next = self.s1(x1, h_s1)
            t1_x1, h_t1_next = self.t1(x1, h_t1)
            y2 = (x2 - t1_x1) / self.e(s1_x1)
            
            s2_y2, h_s2_next = self.s2(y2, h_s2)
            t2_y2, h_t2_next = self.t2(y2, h_t2)
            y1 = (x1 - t2_y2) / self.e(s2_y2)
            
        return [torch.cat((y1, y2), 1)], h_s1_next, h_t1_next, h_s2_next, h_t2_next
    
    def forward(self, x, rev=False):
        # x is a list of tensors, so set x[0]
#         T, batch_size, x_dim = x.shape
        batch_size, T, x_dim = x[0].shape
#         print("x[0].shape: ", x[0].shape)
        
        # initial hidden states for t=0
        h_s1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_t1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_s2 = torch.zeros([batch_size, self.split_len2]).cuda()
        h_t2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
        #
        outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        # print("x[0].is_cuda: ", x[0].is_cuda, x[0].shape)
        # print("x[0][:,0,:].is_cuda: ", x[0][:,0,:].is_cuda)
        # print("x[0][:,1,:].is_cuda: ", x[0][:,1,:].is_cuda)
        # print("x[0][:,2,:].is_cuda: ", x[0][:,2,:].is_cuda)
        # print('---------------------------------------')
        for t in range(T):
            x_t = x[0][:,t,:]
            # print("x_t.is_cuda: ", x_t.is_cuda)
            
            output, h_s1, h_t1, h_s2, h_t2 = self.recurrent_block(x_t, h_s1, h_t1, h_s2, h_t2, rev)
            outputs[:,t,:] = output[0]
        return [outputs]

    def recurrent_block_jacobian(self, x, h_s1, h_t1, h_s2, h_t2, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        
        if not rev:
            s2_x2, h_s2_next = self.s2(x2, h_s2)
            t2_x2, h_t2_next = self.t2(x2, h_t2)
            y1 = self.e(s2_x2) * x1 + t2_x2
            
            s1_y1, h_s1_next = self.s1(y1, h_s1)
            t1_y1, h_t1_next = self.t1(y1, h_t1)
            
            jac = self.log_e(s1_y1) + self.log_e(s2_x2)
        else:
            s1_x1, h_s1_next = self.s1(x1, h_s1)
            t1_x1, h_t1_next = self.t1(x1, h_t1)
            y2 = (x2 - t1_x1) / self.e(s1_x1)
            
            s2_y2, h_s2_next = self.s2(y2, h_s2)
            t2_y2, h_t2_next = self.t2(y2, h_t2)
            
            jac = -self.log_e(s1_x1) - self.log_e(s2_y2)

        return torch.sum(jac, dim=tuple(range(1, self.ndims+1))), h_s1_next, h_t1_next, h_s2_next, h_t2_next 
    
    def jacobian(self, x, rev=False):
        # print('x.shape: ', x[0].shape)

        batch_size,T, x_dim = x[0].shape
        
        # initial hidden states for t=0
        h_s1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_t1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_s2 = torch.zeros([batch_size, self.split_len2]).cuda()
        h_t2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
        #
        outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        for t in range(T):
            x_t = x[0][:,t,:]
            # print('x_t.shape: ', x_t.shape)
            
            output, h_s1, h_t1, h_s2, h_t2 = self.recurrent_block_jacobian(x_t, h_s1, h_t1, h_s2, h_t2, rev)
            outputs[:,t,:] = output[0]
            
        # accumulte sum along dim = 2    
        return torch.sum(outputs, dim=2)

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

class glow_coupling_layer_T(nn.Module):
    def __init__(self, dims_in, F_class=F_fully_connected, F_args={},
                 clamp=5.):
        super(glow_coupling_layer_T, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2*2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1*2, **F_args)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)
    
    def recurrent_block(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 =  self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
        else:  # names of x and y are swapped!
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            
        return [torch.cat((y1, y2), 1)]
    
    def forward(self, x, rev=False):
        # x is a list of tensors, so set x[0]
        batch_size, T, x_dim = x[0].shape
        
        outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
        for t in range(T):
            x_t = x[0][:,t,:]
            output = self.recurrent_block(x_t, rev)
            outputs[:,t,:] = output[0]
        return [outputs]
    
    def recurrent_block_jacobian(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        
        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
        else:
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

        jac = (torch.sum(self.log_e(s1), dim=1)
               + torch.sum(self.log_e(s2), dim=1))
        if rev:
            jac = -jac

        return jac
 
    def jacobian(self, x, rev=False):
        batch_size,T, x_dim = x[0].shape
        
        outputs = torch.zeros(batch_size, T).cuda()
        # x is a tensor: T x batch_size x x_dim
        for t in range(T):
            x_t = x[0][:,t,:]
            
            output = self.recurrent_block_jacobian(x_t, rev)
            outputs[:,t] = output
            
        return outputs
    
    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class glow_gru_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class=F_GRU, F_args={},
                 clamp=5.):
        super(glow_gru_coupling_layer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2*2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1*2, **F_args)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)
    
# Single recurrent step through t. Explicitly output hidden states for t+1.
    def recurrent_block(self, x, h_r1, h_r2, rev=False):
        # print("x.shape in recurrent_block: ", x.shape)
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
#         print("x1.shape in recurrent_block: ", x1.shape)
#         print("x2.shape in recurrent_block: ", x2.shape)
#         print("h_s1.shape: ", h_s1.shape)

        if not rev:
            # print(x2.is_cuda, h_s2.is_cuda)
            r2, h_r2_next = self.s2(x2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2
            

            r1, h_r1_next = self.s1(y1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
        else:  # names of x and y are swapped!
            r1, h_r1_next = self.s1(x1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2, h_r2_next = self.s2(y2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            
        return [torch.cat((y1, y2), 1)], h_r1_next, h_r2_next
    
    def forward(self, x, rev=False):
        # x is a list of tensors, so set x[0]
#         T, batch_size, x_dim = x.shape
        batch_size, T, x_dim = x[0].shape
#         print("x[0].shape: ", x[0].shape)
        
        # initial hidden states for t=0
        h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_r2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
        #
        outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        # print("x[0].is_cuda: ", x[0].is_cuda, x[0].shape)
        # print("x[0][:,0,:].is_cuda: ", x[0][:,0,:].is_cuda)
        # print("x[0][:,1,:].is_cuda: ", x[0][:,1,:].is_cuda)
        # print("x[0][:,2,:].is_cuda: ", x[0][:,2,:].is_cuda)
        # print('---------------------------------------')
        for t in range(T):
            x_t = x[0][:,t,:]
            output, h_r1, h_r2 = self.recurrent_block(x_t, h_r1, h_r2, rev)
            outputs[:,t,:] = output[0]
        return [outputs]
    
    def recurrent_block_jacobian(self, x, h_r1, h_r2, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        
        if not rev:
            r2, h_r2_next = self.s2(x2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1, h_r1_next = self.s1(y1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
        else:
            r1, h_r1_next = self.s1(x1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2, h_r2_next = self.s2(y2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

        jac = (torch.sum(self.log_e(s1), dim=1)
               + torch.sum(self.log_e(s2), dim=1))
        for i in range(self.ndims-1):
            jac = torch.sum(jac, dim=1)

        return jac, h_r1_next, h_r2_next
 
    def jacobian(self, x, rev=False):
        # print('x.shape: ', x[0].shape)

        batch_size,T, x_dim = x[0].shape
        
        # initial hidden states for t=0
        h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_r2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
        #
        outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        for t in range(T):
            x_t = x[0][:,t,:]
            # print('x_t.shape: ', x_t.shape)
            
            output, h_r1, h_r2 = self.recurrent_block_jacobian(x_t, h_r1, h_r2, rev)
            outputs[:,t,:] = output[0]
            
        # accumulte sum along dim = 2    
        return torch.sum(outputs, dim=2)   
    
    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
    
class glow_gru_cgate_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class=F_GRU, F_args={},
                 clamp=5.):
        super(glow_gru_cgate_coupling_layer, self).__init__()
        channels = dims_in[0][0]

        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2*2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1*2, **F_args)
        
#         self.cg1 = F_cgate(self.split_len1)
#         self.cg2 = F_cgate(self.split_len2)

        self.fc1 = nn.Linear(self.split_len1, self.split_len1)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(self.split_len2, self.split_len2)
        self.sig2 = nn.Sigmoid()

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)
    
# Single recurrent step through t. Explicitly output hidden states for t+1.
    def recurrent_block(self, x, h_r1, h_r2, rev=False):
        # print("x.shape in recurrent_block: ", x.shape)
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
#         print("x1.shape in recurrent_block: ", x1.shape)
#         print("x2.shape in recurrent_block: ", x2.shape)
#         print("h_s1.shape: ", h_s1.shape)

        if not rev:
            # We can also intentionally use h_r1
            
            # print(x2.is_cuda, h_s2.is_cuda)
            r2, h_r2_next = self.s2(x2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2
            
            x2 = x2 * self.sig2(self.fc2(h_r1))

            r1, h_r1_next = self.s1(y1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            
            y1 = y1 * self.sig1(self.fc1(h_r2))
        else:  # names of x and y are swapped!   
            
            r1, h_r1_next = self.s1(x1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)  
            
            x1 = x1 * self.sig1(self.fc1(h_r2))
            
            r2, h_r2_next = self.s2(y2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            
            y2 = y2 * self.sig2(self.fc2(h_r1))
            
        return torch.cat((y1, y2), 1), h_r1_next, h_r2_next
    
    def forward(self, x, rev=False):
        # x is a list of tensors, so set x[0]
#         T, batch_size, x_dim = x.shape
        batch_size, T, x_dim = x[0].shape
#         print("x[0].shape: ", x[0].shape)
        
        # initial hidden states for t=0
        h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_r2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
        #
        outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        # print("x[0].is_cuda: ", x[0].is_cuda, x[0].shape)
        # print("x[0][:,0,:].is_cuda: ", x[0][:,0,:].is_cuda)
        # print("x[0][:,1,:].is_cuda: ", x[0][:,1,:].is_cuda)
        # print("x[0][:,2,:].is_cuda: ", x[0][:,2,:].is_cuda)
        # print('---------------------------------------')
        for t in range(T):
            x_t = x[0][:,t,:]
            output, h_r1, h_r2 = self.recurrent_block(x_t, h_r1, h_r2, rev)
            outputs[:,t,:] = output[0]
        return [outputs]
    
    def recurrent_block_jacobian(self, x, h_r1, h_r2, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        
        if not rev:
            r2, h_r2_next = self.s2(x2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1, h_r1_next = self.s1(y1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
        else:
            r1, h_r1_next = self.s1(x1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2, h_r2_next = self.s2(y2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

        jac = (torch.sum(self.log_e(s1), dim=1)
               + torch.sum(self.log_e(s2), dim=1)
              + torch.sum(self.sig1(self.fc1(h_r2)), dim=1)
              + torch.sum(self.sig2(self.fc2(h_r1)), dim=1))
        for i in range(self.ndims-1):
            jac = torch.sum(jac, dim=1)

        return jac, h_r1_next, h_r2_next
 
    def jacobian(self, x, rev=False):
        # print('x.shape: ', x[0].shape)

        batch_size,T, x_dim = x[0].shape
        
        # initial hidden states for t=0
        h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
        h_r2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
        #
        outputs = torch.zeros(batch_size, T).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        for t in range(T):
            x_t = x[0][:,t,:]
            # print('x_t.shape: ', x_t.shape)
            
            output, h_r1, h_r2 = self.recurrent_block_jacobian(x_t, h_r1, h_r2, rev)
            outputs[:,t] = output
            
        # accumulte sum along dim = 2    
        return outputs
    
    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

class glow_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class=F_fully_connected, F_args={},
                 clamp=5.):
        super(glow_coupling_layer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2*2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1*2, **F_args)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]

        else:  # names of x and y are swapped!
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

        jac = (torch.sum(self.log_e(s1), dim=1)
               + torch.sum(self.log_e(s2), dim=1))
        for i in range(self.ndims-1):
            jac = torch.sum(jac, dim=1)

        return jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

class glow_gru_residual_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class=F_GRU_BN,F_cgate=F_cgate, F_args={},
                 clamp=5.):
        super(glow_gru_residual_coupling_layer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2*2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1*2, **F_args)
        
        '''structure for gating '''
        self.cg1 = F_cgate(self.split_len1)
        self.cg2 = F_cgate(self.split_len2)
        
    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)
    
# Single recurrent step through t. Explicitly output hidden states for t+1.
    def recurrent_block(self, x, h_r1, h_r2, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            # We can also intentionally use h_r1
            
            r2, h_r2_next = self.s2(x2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2
            
            x2 = x2 * self.e(self.cg1(h_r1))

            r1, h_r1_next = self.s1(y1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            
            y1 = y1 * self.e(self.cg2(h_r2))
        else:  # names of x and y are swapped!   
            
            x1 = x1 / self.e(self.cg2(h_r2))

            r1, h_r1_next = self.s1(x1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1) 
            
            
            y2 = y2 / self.e(self.cg1(h_r1))

            r2, h_r2_next = self.s2(y2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            
            
        return [torch.cat((y1, y2), 1)], h_r1_next, h_r2_next
    
    def forward(self, x, rev=False):
        # x is a list of tensors, so set x[0]
#         T, batch_size, x_dim = x.shape
        batch_size, T, x_dim = x[0].shape
#         print("x[0].shape: ", x[0].shape)
        
        if not rev:
            # initial hidden states for t=0
            h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
            h_r2 = torch.zeros([batch_size, self.split_len1]).cuda()
        else:
            # initial hidden states for t=0
            h_r1 = torch.ones([batch_size, self.split_len1]).cuda()
            h_r2 = torch.ones([batch_size, self.split_len1]).cuda()
        
        #
        outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        # print("x[0].is_cuda: ", x[0].is_cuda, x[0].shape)
        # print("x[0][:,0,:].is_cuda: ", x[0][:,0,:].is_cuda)
        # print("x[0][:,1,:].is_cuda: ", x[0][:,1,:].is_cuda)
        # print("x[0][:,2,:].is_cuda: ", x[0][:,2,:].is_cuda)
        # print('---------------------------------------')
        for t in range(T):
            x_t = x[0][:,t,:]
            output, h_r1, h_r2 = self.recurrent_block(x_t, h_r1, h_r2, rev)
            outputs[:,t,:] = output[0]
        return [outputs]
    
    def recurrent_block_jacobian(self, x, h_r1, h_r2, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        
        if not rev:
            r2, h_r2_next = self.s2(x2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1, h_r1_next = self.s1(y1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
        else:
            r1, h_r1_next = self.s1(x1, h_r1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2, h_r2_next = self.s2(y2, h_r2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

        jac = (torch.sum(self.log_e(s1), dim=1)
               + torch.sum(self.log_e(s2), dim=1)
               + torch.sum(self.log_e(self.cg1(h_r1)), dim=1)
               + torch.sum(self.log_e(self.cg2(h_r2)), dim=1))
        if rev: 
            jac = -jac 

        return jac, h_r1_next, h_r2_next
 
    def jacobian(self, x, rev=False):
        # print('x.shape: ', x[0].shape)

        batch_size,T, x_dim = x[0].shape
        
        if not rev:
            # initial hidden states for t=0
            h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
            h_r2 = torch.zeros([batch_size, self.split_len1]).cuda()
        else:
            # initial hidden states for t=0
            h_r1 = torch.ones([batch_size, self.split_len1]).cuda()
            h_r2 = torch.ones([batch_size, self.split_len1]).cuda()
        
        outputs = torch.zeros(batch_size, T).cuda()
        
        # x is a tensor: T x batch_size x x_dim
        for t in range(T):
            x_t = x[0][:,t,:]
            
            output, h_r1, h_r2 = self.recurrent_block_jacobian(x_t, h_r1, h_r2, rev)
            outputs[:,t] = output
            
        return outputs
    
    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

# class glow_gru_residual_coupling_layer_one_gru(nn.Module):
#     def __init__(self, dims_in, F_class=F_GRU_BN,F_cgate=F_cgate, F_args={},
#                  clamp=5.):
#         super(glow_gru_residual_coupling_layer, self).__init__()
#         channels = dims_in[0][0]
#         self.ndims = len(dims_in[0])
#         self.split_len1 = channels // 2
#         self.split_len2 = channels - channels // 2
        # assert self.split_len1 == self.split_len2 
#         self.clamp = clamp
#         self.max_s = exp(clamp)
#         self.min_s = exp(-clamp)

#         self.s = F_class(self.split_len1, self.split_len2*2, **F_args)
#         # self.s2 = F_class(self.split_len2, self.split_len1*2, **F_args)
        
#         '''structure for gating '''
#         self.cg = F_cgate(self.split_len1)
#         # self.cg2 = F_cgate(self.split_len2)
        
#         self.fc = nn.Linear(self.split_len1, self.split_len1)
#         self.sig = nn.Sigmoid()
#         # self.fc2 = nn.Linear(self.split_len2, self.split_len2)
#         # self.sig2 = nn.Sigmoid()

#     def e(self, s):
#         return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

#     def log_e(self, s):
#         return self.clamp * 0.636 * torch.atan(s / self.clamp)
    
# # Single recurrent step through t. Explicitly output hidden states for t+1.
#     def recurrent_block(self, x, h_r, rev=False):
#         x1, x2 = (x.narrow(1, 0, self.split_len1),
#                   x.narrow(1, self.split_len1, self.split_len2))

#         if not rev: 
#             r, h_r_next  = self.s(x2, h_r)
#             s, t = r[:, :self.split_len2], r[:, self.split_len2:]
#             y1 = self.e(s) * x1 + t

#             x2 = x2 * self.e(self.cg(h_r_next))

#             # permute 
#             r, h_r_next_next  = self.s(y1, h_r_next)
#             s, t = r[:, :self.split_len1], r[:, self.split_len1:]
#             y2 = self.e(s) * x2 + t

#             y1 = y1 * self.e(self.cg(h_r_next_next))

#         else: # names of x and y are swapped!   
#             x1 = x1 / self.e(self.cg(h_r))
#             r, h_r_prev = self.s(x1, h_r)
#             s, t = r[:, :self.split_len2], r[:, self.split_len2:]
#             y2 = (x2 - t) / self.e(s)
            
            # y2 = y2 / self.e(self.cg(h_r_prev))
            # r, h_r_prev_prev = self.s(y2, h_r_prev)


#         return torch.cat((y1, y2), 1), h_r_next_next

#     def recurrent_block(self, x, h_r1, h_r2, rev=False):
#         x1, x2 = (x.narrow(1, 0, self.split_len1),
#                   x.narrow(1, self.split_len1, self.split_len2))

#         if not rev:
#             # We can also intentionally use h_r1
            
#             r2, h_r2_next = self.s2(x2, h_r2)
#             s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
#             y1 = self.e(s2) * x1 + t2
            
#             x2 = x2 * self.sig2(self.fc2(h_r1))

#             r1, h_r1_next = self.s1(y1, h_r1)
#             s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
#             y2 = self.e(s1) * x2 + t1
            
#             y1 = y1 * self.sig1(self.fc1(h_r2))
#         else:  # names of x and y are swapped!   
            
#             x1 = x1 / self.sig1(self.fc1(h_r2))
#             r1, h_r1_next = self.s1(x1, h_r1)
#             s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
#             y2 = (x2 - t1) / self.e(s1)  
            
            
#             y2 = y2 / self.sig2(self.fc2(h_r1))
#             r2, h_r2_next = self.s2(y2, h_r2)
#             s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
#             y1 = (x1 - t2) / self.e(s2)
            
            
#         return torch.cat((y1, y2), 1), h_r1_next, h_r2_next
    
#     def forward(self, x, rev=False):
#         # x is a list of tensors, so set x[0]
# #         T, batch_size, x_dim = x.shape
#         batch_size, T, x_dim = x[0].shape
# #         print("x[0].shape: ", x[0].shape)
        
#         # initial hidden states for t=0
#         h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
#         h_r2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
#         #
#         outputs = torch.zeros(batch_size, T, x_dim).cuda()
        
#         # x is a tensor: T x batch_size x x_dim
#         # print("x[0].is_cuda: ", x[0].is_cuda, x[0].shape)
#         # print("x[0][:,0,:].is_cuda: ", x[0][:,0,:].is_cuda)
#         # print("x[0][:,1,:].is_cuda: ", x[0][:,1,:].is_cuda)
#         # print("x[0][:,2,:].is_cuda: ", x[0][:,2,:].is_cuda)
#         # print('---------------------------------------')
#         for t in range(T):
#             x_t = x[0][:,t,:]
#             output, h_r1, h_r2 = self.recurrent_block(x_t, h_r1, h_r2, rev)
#             outputs[:,t,:] = output[0]
#         return [outputs]
    
#     def recurrent_block_jacobian(self, x, h_r1, h_r2, rev=False):
#         x1, x2 = (x.narrow(1, 0, self.split_len1),
#                   x.narrow(1, self.split_len1, self.split_len2))
        
#         if not rev:
#             r2, h_r2_next = self.s2(x2, h_r2)
#             s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
#             y1 = self.e(s2) * x1 + t2

#             r1, h_r1_next = self.s1(y1, h_r1)
#             s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
#         else:
#             r1, h_r1_next = self.s1(x1, h_r1)
#             s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
#             y2 = (x2 - t1) / self.e(s1)

#             r2, h_r2_next = self.s2(y2, h_r2)
#             s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

#         jac = (torch.sum(self.log_e(s1), dim=1)
#                + torch.sum(self.log_e(s2), dim=1)
#                + torch.sum(self.sig1(self.fc1(h_r2)), dim=1)
#                + torch.sum(self.sig2(self.fc2(h_r1)), dim=1))
#         for i in range(self.ndims-1):
#             jac = torch.sum(jac, dim=1)

#         return jac, h_r1_next, h_r2_next
 
#     def jacobian(self, x, rev=False):
#         # print('x.shape: ', x[0].shape)

#         batch_size,T, x_dim = x[0].shape
        
#         # initial hidden states for t=0
#         h_r1 = torch.zeros([batch_size, self.split_len1]).cuda()
#         h_r2 = torch.zeros([batch_size, self.split_len2]).cuda()
        
#         #
#         outputs = torch.zeros(batch_size, T).cuda()
        
#         # x is a tensor: T x batch_size x x_dim
#         for t in range(T):
#             x_t = x[0][:,t,:]
            
#             output, h_r1, h_r2 = self.recurrent_block_jacobian(x_t, h_r1, h_r2, rev)
#             outputs[:,t] = output
            
#         return outputs
    
#     def output_dims(self, input_dims):
#         assert len(input_dims) == 1, "Can only use 1 input"
#         return input_dims