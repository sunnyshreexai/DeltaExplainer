"""Module pointing to different implementations of Data class
"""

from data_interfaces.base_data_interface import _BaseData


class Data(_BaseData):
    """Class containing all required information about the data for DiCE."""

    def __init__(self, **params):
        """Init method

        :param **params: a dictionary of required parameters.
        """
        self.decide_implementation_type(params)

    def decide_implementation_type(self, params):
        """Decides if the Data class is for public or private data."""
        self.__class__ = decide(params)
        self.__init__(params)


def decide(params):
    """Decides if the Data class is for public or private data.

    To add new implementations of Data, add the class in data_interfaces
    subpackage and import-and-return the class in an elif loop as shown
    in the below method.
    """
    if 'dataframe' in params:
        # if params contain a Pandas dataframe, then use PublicData class
        from data_interfaces.public_data_interface import PublicData
        return PublicData
    else:
        # use PrivateData if only meta data is provided
        from data_interfaces.private_data_interface import PrivateData
        return PrivateData
