import torch
import numpy as np
import torch.nn.functional as F

from dolphin.base.base_model_module import BaseModelModule
from dolphin.utils import Registers, Bar


@Registers.strategy.register
class LeastConfidenceSampling(BaseModelModule):

    def __init__(self, **kwargs):

        super(LeastConfidenceSampling, self).__init__(**kwargs)
    
    def init_weights(self):
        pass

    def forward(self):
        pass

    def query(self, model, data_loader, logger=None):
        probs = self.predict(model, data_loader, output_format='prob', logger=logger)
        uncertainty = probs.max(1)[0]
        unlabeled = ~data_loader.dataset.idx_labeled
        idx_unlabeled = np.where(unlabeled == True)
        return idx_unlabeled[uncertainty.sort()[1][:self.num_query]]
    
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