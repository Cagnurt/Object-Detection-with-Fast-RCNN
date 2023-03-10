import torch.nn.functional as F

class Config:
    def __init__(self,name):
        self.model = None
        self.optimizer = None
        self.criterion = F.nll_loss
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.log_freq = 1000
        self.epoch = 15
        self.hidden_dim = 64
        self.sweep = {  "name" : "Stroma Challange",
                        "method" : "grid",
                        "parameters" : {
                            "learning_rate" : {
                                "values" : [0.01, 0.001]
                            },
                            "momentum" :{
                                "values" : [0.1, 0.5, 0.9]
                            },
                        }
                      }