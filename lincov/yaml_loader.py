import numpy as np

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
yaml = YAML()


class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def scale(val, scale):
    if type(val) in (CommentedSeq, list, tuple):
        return np.array(val) * scale
    else:
        return val * scale
        
    
class YamlLoader(object):
    required_params = ('horizon_fov',
                       'horizon_max_phase_angle',
                       'radiometric_min_elevation')
                       
    
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
                self.params[key[:-4]] = scale(self.params[key], np.pi/180.0)
            elif key[-7:] == '_arcmin':
                self.params[key[:-7]] = scale(self.params[key], np.pi / (180*60.0))
            elif key[-7:] == '_arcsec':
                self.params[key[:-7]] = scale(self.params[key], np.pi / (180*3600.0))
            elif type(self.params[key]) == CommentedSeq:
                self.params[key] = np.array(self.params[key])

        # Set defaults for mandatory arguments. These are only here
        # because there are computations in State which happen
        # regardless of the measurement types that depend on these
        # parameters. State does not have access to the meas_dt dict.
        if 'horizon_fov' not in self.params:
            print("Setting default: horizon_fov = 0")
            self.params.horizon_fov = 0.0
        if 'horizon_max_phase_angle' not in self.params:
            print("Setting default: horizon_max_phase_angle = 0")
            self.params.horizon_max_phase_angle = 0.0
        if 'radiometric_min_elevation' not in self.params:
            print("Setting default: radiometric_min_elevation_deg = 90")
            self.params.radiometric_min_elevation = np.pi * 0.5 # 90 degrees
                
    def as_metadata(self):
        metadata              = self.yaml
        metadata['order']     = self.order
        metadata['meas_last'] = self.meas_last
        metadata['meas_dt']   = self.meas_dt
        return metadata
