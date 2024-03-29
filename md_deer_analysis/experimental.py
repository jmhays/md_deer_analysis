from abc import ABC
import json


class MetaData(ABC):

    def __init__(self, name):
        """
        Construct metadata object and give it a name
        :param name: All metadata classes should have names that associate them with a particular pair.
        """
        self.__name = name
        self.__required_parameters = []
        self._metadata = {}

    @property
    def name(self):
        return self.__name

    @name.getter
    def name(self):
        return self.__name

    def set_requirements(self, list_of_requirements: list):
        self.__required_parameters = list_of_requirements

    def get_requirements(self):
        return self.__required_parameters

    def set(self, key, value):
        self._metadata[key] = value

    def get(self, key):
        return self._metadata[key]

    def set_from_dictionary(self, data):
        self._metadata = data

    def get_as_dictionary(self):
        return self._metadata

    def get_missing_keys(self):
        missing = []
        for required in self.__required_parameters:
            if required not in self._metadata.keys():
                missing.append(required)
        return missing


class MultiMetaData(ABC):

    def __init__(self):
        self._metadata_list = []
        self._names = []

    def get_names(self):
        if not self._names:
            if not self._metadata_list:
                raise IndexError('Must import a list of metadata before retrieving names')
            self._names = [metadata.name for metadata in self._metadata_list]
        return self._names

    def set_names(self, names):
        for name in names:
            self._names.append(name)

    def name_to_id(self, name):
        if not self._names:
            _ = self.get_names()
        return self._names.index(name)

    def id_to_name(self, id):
        return self._names[id]

    def __getitem__(self, item):
        return self._metadata_list[item]

    def __setitem__(self, key, value):
        self._metadata_list[key]=value

    def __delitem__(self, key):
        self._metadata_list.__delitem__(key)

    def __sizeof__(self):
        return len(self._metadata_list)

    def get_as_single_dataset(self):
        single_dataset = {}
        for metadata in self._metadata_list:
            single_dataset[metadata.name] = metadata.get_as_dictionary()
        return single_dataset

    def write_to_json(self, filename='state.json'):
        json.dump(self.get_as_single_dataset(), open(filename, 'w'))

    def read_from_json(self, filename='state.json'):
        # TODO: decide on expected behavior here if there's a pre-existing list of data. For now, overwrite
        self._metadata_list = []
        self._names = []
        data = json.load(open(filename, 'r'))
        for name, metadata in data.items():
            self._names.append(name)
            metadata_obj = MetaData(name=name)
            metadata_obj.set_from_dictionary(metadata)
            self._metadata_list.append(metadata_obj)


class PairData(MetaData):
    def __init__(self, name):
        super().__init__(name=name)
        self.set_requirements(['distribution', 'bins', 'sites'])


class MultiPair(MultiMetaData):
    def __init__(self):
        super().__init__()
        self.num_pairs = 0

    def read_from_json(self, filename='state.json'):
        self._metadata_list = []
        self._names = []
        data = json.load(open(filename, 'r'))
        for name, metadata in data.items():
            self._names.append(name)
            metadata_obj = PairData(name=name)
            metadata_obj.set_from_dictionary(metadata)
            self._metadata_list.append(metadata_obj)

        self.num_pairs = len(self._names)
