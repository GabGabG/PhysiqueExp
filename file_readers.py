import re
import numpy
from abc import ABC


def get_reader(file_name):
    if file_name.endswith(".mca"):
        reader = Mca_file_reader(file_name)
    elif file_name.endswith(".Spe"):
        reader = Spe_file_reader(file_name)
    else:
        raise FileExtensionError("This file type is not supported.")
    return reader


class FileExtensionError(Exception):
    pass


class file_reader(ABC):
    def __init__(self, file_name=None):
        if file_name is not None:
            self.histogram_data = None
            self.read(file_name)

    def read(self, file_name):
        assert isinstance(file_name, str), "The input must be a string"
        self.verify_file_extension(file_name)
        with open(file_name, 'r', errors="ignore") as file_handler:
            self.file_data = file_handler.readlines()
        self.find_histogram_data()
        self.find_header_information()
        self.normalize_data()

    def find_histogram_data(self):
        pass

    def find_header_information(self):
        pass

    def verify_file_extension(self, file_name):
        pass

    def get_x_indices(self):
        assert self.histogram_data is not None, "You must have histogram data"
        return numpy.arange(len(self.histogram_data))

    def get_slice(self, minimum, maximum):
        assert self.histogram_data is not None, "You must have histogram data"
        x_data = self.get_x_indices()[minimum:maximum]
        y_data = self.histogram_data[minimum:maximum]
        return [x_data, y_data]

    def normalize_data(self):
        self.histogram_data = self.histogram_data / numpy.max(self.histogram_data)


class Mca_file_reader(file_reader):

    def find_histogram_data(self):
        first_element = self.file_data.index("<<DATA>>\n") + 1
        last_element = self.file_data.index("<<END>>\n")
        self.histogram_data = self.file_data[first_element:last_element]
        self.histogram_data = numpy.array(self.histogram_data).astype(numpy.int32)

    def find_header_information(self):
        first_element = self.file_data.index("<<DPP CONFIGURATION>>\n") + 1
        last_element = self.file_data.index("<<DPP CONFIGURATION END>>\n")
        self.header_information = {}
        header = self.file_data[first_element:last_element]
        for line in header:
            splitted = line.split(sep=':')
            tag = splitted[0]
            value = ' '.join(splitted[1:]).rstrip('\n')
            self.header_information[tag] = value

    def verify_file_extension(self, file_name):
        mca_extension_regex = re.compile(".mca$")
        if mca_extension_regex.match(file_name) is not None:
            raise FileExtensionError("The file name needs have "
                                     "the .mca extension.")


class Spe_file_reader(file_reader):

    def find_histogram_data(self):
        first_element = self.file_data.index("$DATA:\n") + 2
        last_element = self.file_data.index("$ROI:\n")
        self.histogram_data = self.file_data[first_element:last_element]
        self.histogram_data = numpy.array(self.histogram_data).astype(numpy.int32)

    def find_header_information(self):
        self.header_information = None

    def verify_file_extension(self, file_name):
        mca_extension_regex = re.compile(".Spe$")
        if mca_extension_regex.match(file_name) is not None:
            raise FileExtensionError("The file name needs have "
                                     "the .Spe extension.")
