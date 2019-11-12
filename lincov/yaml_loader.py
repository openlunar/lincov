import numpy as np

from ruamel.yaml import YAML
yaml = YAML()


class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
class YamlLoader(object):
    def __init__(self, label):
        f = open("config/{}.yml".format(label), 'r')
        self.yaml = yaml.load(f)
        self.dt = self.yaml['dt']
        self.order = list(self.yaml['meas_dt'].keys())
        self.meas_dt = {}
        self.meas_last = {}
        self.block_dt = self.yaml['block_dt']
        
        for key in self.order:
            self.meas_dt[key] = self.yaml['meas_dt'][key]
            
        if 'meas_last' in self.yaml:
            for key in self.yaml['meas_last']:
                self.meas_last[key] = self.yaml['meas_last'][key]

        for key in self.order:
            if key not in self.meas_last:
                self.meas_last[key] = 0.0
                
        self.label = label
        self.params = AttributeDict(self.yaml['params'])

        # Perform unit conversions for keys
        for key in self.yaml['params']:
            if key[-4:] == '_deg':
                self.params[key[:-4]] = self.params[key] * np.pi/180.0

    def as_metadata(self):
        metadata              = self.yaml
        metadata['order']     = self.order
        metadata['meas_last'] = self.meas_last
        metadata['meas_dt']   = self.meas_dt
        return metadata
