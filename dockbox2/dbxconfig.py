import os 
import sys
import configparser

default_options = {'GENERAL': {'epochs': {'required': True, 'type': int},
                                       'depth': {'default': 2, 'type': int},
                                       'activation': {'default': 'sigmoid'},
                                       'nrof_neigh': {'default': 25, 'type': int},
                                       'edge_feature': {'default': None}},

'MINIBATCH': {'batch_size': {'default': 2, 'type': int},
              'num_parallel_calls': {'default': 1, 'type': int}},

'LOSSN': {'type': {'default': 'BinaryFocalCrossentropy'},
         'apply_class_balancing': {'default': False, 'type': bool, 'with_option': ('type', 'BinaryFocalCrossentropy')},
         'alpha': {'default': 0.25, 'type': float, 'with_option': ('type', 'BinaryFocalCrossentropy')},
         'gamma': {'default': 2.0, 'type': float, 'with_option': ('type', 'BinaryFocalCrossentropy')},
         'weight': {'default': 1.0, 'type': float}},

'LOSSR': {'weight': {'default': 1.0, 'type': float}},

'AGGREGATOR': {'shape': {'required': True, 'type': int},
              'type': {'default': 'pooling'},
              'use_concat': {'default': True, 'type': bool},
              'activation': {'default': 'leaky_relu'}},

'ATTENTION': { 'shape': {'default': None, 'type': int},
               'activation': {'default': 'leaky_relu'}}
}

default_options['LOSSG'] = default_options['LOSSN']

class ConfigSetup(object):

    def __init__(self, inifile):

        self.inifile = inifile
        if not os.path.isfile(inifile):
            raise IOError("File %s does not exist!"%inifile)

        self.load_parameters(inifile)

    def load_parameters(self, inifile):

        config = configparser.SafeConfigParser()
        config.read(inifile)

        # check if required options have been set
        for section in default_options:
            for option, properties in default_options[section].items():

                if 'required' in properties and properties['required']:
                    if not section in config.sections():
                        sys.exit("section %s is mandatory in .ini configuration file!"%section)

                    if not config.has_option(section, option):
                        sys.exit("option %s in section %s is mandatory in .ini configuration file!"%(option, section))

        parameters = {}
        # store settings in parameters
        for section in config.sections():
            if section not in default_options:
                sys.exit("section %s not recognized in .ini configuration file!"%section)
            else:
                options = dict(config.items(section))
                for option in options:
                   if option not in default_options[section]:
                       sys.exit("option %s in section %s not recognized in .ini configuration file"%(option, section))

                   else:
                       if section not in parameters:
                           parameters[section] = {}

                       if 'type' in default_options[section][option]:
                           converter = default_options[section][option]['type']
                           value = converter(config.get(section, option))
                       else:
                           value = str(config.get(section, option).strip())
                       parameters[section][option] = value

        # set default options if they have not been set
        for section in default_options:
            for option, properties in default_options[section].items():
                if 'default' in properties:
                    if section not in parameters:
                        parameters[section] = {}

                    if option not in parameters[section]:
                        parameters[section][option] = properties['default']

        # remove unrelated options
        for section in default_options:
            for option, properties in default_options[section].items():
                if 'with_option' in properties:
                    related_option, value = properties['with_option']

                    if parameters[section][related_option] != value:
                        parameters[section].pop(option)

        # set general options as direct attributes
        self.epochs = parameters['GENERAL']['epochs']
        self.depth = parameters['GENERAL']['depth']

        self.activation = parameters['GENERAL']['activation']
        self.nrof_neigh = parameters['GENERAL']['nrof_neigh']
        self.edge_feature = parameters['GENERAL']['edge_feature']

        self.minibatch = parameters['MINIBATCH']
        self.loss = {'loss_n': parameters['LOSSN'], 
                     'loss_g': parameters['LOSSG'],
                     'loss_reg': parameters['LOSSR']}

        self.aggregator = parameters['AGGREGATOR']
        self.attention = parameters['ATTENTION']
