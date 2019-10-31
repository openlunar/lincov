from ruamel.yaml import YAML
yaml = YAML()

class YamlLoader(object):
    def __init__(self, label):
        f = open("config/{}.yml".format(label), 'r')
        self.yaml = yaml.load(f)
        self.dt = self.yaml['dt']
        self.order = list(self.yaml['order'])
        self.meas_dt = {}
        self.block_dt = self.yaml['block_dt']
        for key in self.yaml['meas_dt']:
            self.meas_dt[key] = self.yaml['meas_dt'][key]
        self.label = label
        self.params = self.yaml['params']
