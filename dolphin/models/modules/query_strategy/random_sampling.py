import torch
import torch.nn.functional as F
import numpy as np

from dolphin.base.base_model_module import BaseModelModule
from dolphin.utils import Registers, Bar


@Registers.strategy.register
class RandomSampling(BaseModelModule):

    def __init__(self, **kwargs):

        super(RandomSampling, self).__init__(**kwargs)

    def init_weights(self):
        pass

    def forward(self):
        pass

    def query(self, model, data_loader):
        idx_labeled = data_loader.dataset.idx_labeled
        return np.random.choice(np.where(idx_labeled == 0)[0], self.num_query)

    def predict(self, model, data_loader, output_format='prob', logger=None):
        assert output_format in ['prob', 'label']
        num_data = len(data_loader)
        num_class = data_loader.dataset.num_classes
        p = []
        
        bar = Bar('Query Mode', max=(num_data))
        
        with torch.no_grad():
            for idx, data_batch in enumerate(data_loader):
                output = model.test_step(data_batch)
                results = output['results']
                if output_format == 'prob':
                    prob = F.softmax(results, dim=1)
                    p.append(prob.cpu())
                else:
                    pred = results.max(1)[1]
                    p.append(prob.cpu())

                print_str = '[{}/{}]'.format(idx + 1, num_data)
                Bar.suffix = print_str
                bar.next()
                
            p = torch.cat(p)
        return p