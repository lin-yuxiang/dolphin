from dolphin.base.base_algorithm import BaseAlgorithm
from dolphin.utils import Registers, build_module_from_registers
from dolphin.models.utils import ImagePool


@Registers.algorithm.register
class CycleGAN(BaseAlgorithm):

    def __init__(self,
                 direction='AtoB',
                 pretrained=None,
                 pretrained_modules=None,
                 generator_a=dict(),
                 generator_b=dict(),
                 discriminator_a=dict(),
                 discriminator_b=dict(),
                 loss_gan=dict(),
                 loss_cycle_A=dict(),
                 loss_cycle_B=dict(),
                 loss_idt=dict(),
                 train_cfg=None, # TODO: pool_size in train_cfg
                 test_cfg=None,
                 **kwargs):
        
        super(CycleGAN, self).__init__(
            pretrained=pretrained,
            pretrained_modules=pretrained_modules)

        self.direction = direction

        self.netG_A = build_module_from_registers(
            generator_a, module_name='generator')

        self.netG_B = build_module_from_registers(
            generator_b, module_name='generator')
        
        self.mode = 'train'
        if self.mode == 'train':
            self.netD_A = build_module_from_registers(
                discriminator_a, module_name='discriminator')
            self.netD_B = build_module_from_registers(
                discriminator_b, module_name='discriminator')

            if loss_idt is not None and loss_idt.get('loss_weight') is not None:
                input_nc = generator_a['in_channels']
                output_nc = generator_a['out_channels']
                assert input_nc == output_nc
            
            pool_size = train_cfg['pool_size']
            self.fake_A_pool = ImagePool(pool_size)
            self.fake_B_pool = ImagePool(pool_size)

            self.loss_gan = build_module_from_registers(
                loss_gan, module_name='loss')

            self.loss_cycle_A = build_module_from_registers(
                loss_cycle_A, module_name='loss')

            self.loss_cycle_B = build_module_from_registers(
                loss_cycle_B, module_name='loss')                
            
            self.loss_idt = build_module_from_registers(
                loss_idt, module_name='loss')
        
        self.init_weights()

    def set_input(self, imgs):
        AtoB = self.direction == 'AtoB'
        real_A = imgs['A' if AtoB else 'B']
        real_B = imgs['B' if AtoB else 'A']
        # img_path = imgs['A_paths' if AtoB else 'B_paths']
        return real_A, real_B

    def forward_generator(self, real_A, real_B):
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)
        return fake_B, rec_A, fake_A, rec_B

    def loss_generator(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):

        if self.loss_idt is not None:
            lambda_idt = self.loss_idt.loss_weight
        else:
            lambda_idt = 0
        if self.loss_cycle_A is not None:
            lambda_A = self.loss_cycle_A.loss_weight
        else:
            lambda_A = 1
        if self.loss_cycle_B is not None:
            lambda_B = self.loss_cycle_B.loss_weight
        else:
            lambda_B = 1
        if lambda_idt > 0:
            idt_A = self.netG_A(real_B)
            idt_B = self.netG_B(real_A)
            loss_idt_A = self.loss_idt(idt_A, real_B) * lambda_B
            loss_idt_B = self.loss_idt(idt_B, real_A) * lambda_A
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        
        loss_G_A = self.loss_gan(self.netD_A(fake_B), True)
        loss_G_B = self.loss_gan(self.netD_B(fake_A), True) 
        loss_cycle_A = self.loss_cycle_A(rec_A, real_A)
        loss_cycle_B = self.loss_cycle_B(rec_B, real_B)

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + \
            loss_idt_A + loss_idt_B
        losses = dict(
            loss_G_A=loss_G_A, loss_G_B=loss_G_B, loss_cycle_A=loss_cycle_A, 
            loss_cycle_B=loss_cycle_B, loss_idt_A=loss_idt_A, 
            loss_idt_B=loss_idt_B)
        return loss_G, losses
    
    def loss_discriminator(self, net_D, real, fake):
        pred_real = net_D(real)
        loss_D_real = self.loss_gan(pred_real, True)
        pred_fake = net_D(fake.detach())
        loss_D_fake = self.loss_gan(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        losses = dict(loss_D_real=loss_D_real, loss_D_fake=loss_D_fake)
        return loss_D, losses

    def forward_train(self, imgs, optimizer, **kwargs):
        optimizer_G = optimizer['G']
        optimizer_D = optimizer['D']
        real_A, real_B = self.set_input(imgs)
        fake_B, rec_A, fake_A, rec_B = self.forward_generator(real_A, real_B)

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        loss_G, losses_G = self.loss_generator(
            real_A, real_B, fake_A, fake_B, rec_A, rec_B)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        optimizer_D.zero_grad()
        fake_B = self.fake_B_pool.query(fake_B)
        loss_D_A, losses_D_A = self.loss_discriminator(
            self.netD_A, real_B, fake_B)
        loss_D_A.backward()
        fake_A = self.fake_A_pool.query(fake_A)
        loss_D_B, losses_D_B = self.loss_discriminator(
            self.netD_B, real_A, fake_A)
        loss_D_B.backward()
        optimizer_D.step()

        losses = dict(loss_G=loss_G, loss_D_A=loss_D_A, loss_D_B=loss_D_B)
        losses.update(losses_G)
        losses.update(losses_D_A)
        losses.update(losses_D_B)
        return losses

    def forward_test(self, imgs, **kwargs):
        real_A, real_B = self.set_input(imgs)
        fake_B, rec_A, fake_A, rec_B = self.forward_generator(real_A, real_B)
        results = dict(
            real_A=real_A, real_B=real_B, fake_A=fake_A, fake_B=fake_B, 
            rec_A=rec_A, rec_B=rec_B)
        return results
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad