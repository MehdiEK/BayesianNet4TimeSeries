"""
Version v0 of the custom model

Creation date: 11/02/2024
Last modification: 11/02/2024
By: Mehdi EL KANSOULI 
"""

import numpy as np 
import pandas as pd 


class DumbDiscretizer(object):
    """
    This class aims to control the discreztization of our data. This class aims 
    to take as initial intput the raw time series and prepare a discretized 
    pandas data frame. 
    """

    def __init__(self, df, nb_classes={}):
        """
        Initialization of the class.

        :params df: pd.DataFrame
            Dataframe to discretize 
        :params nb_classes: dict, default=empty
            Dictionary with columns for keys and nb of classes to make.
            If dictionary is empty, log(n) classes are constructed.
            for values. 
        """
        self.df = df.copy()  # orignal dataframe
        self.nb_classes = nb_classes  # nb of classes to create

        # extrema of time series 
        self.extrema = df.agg(['min', 'max']).to_dict() 

        # disctionary that maps discretization to index
        self._indexer = {}
        self._reverse_indexer = {}

        # fit automatically
        self.__fit__()

    def __fit__(self):
        """
        Function that learns how to map a value of the initial dataframe 
        to an index of the pgmpy object. 
        """
        for column in self.df.columns:
            self.column_indexer(column)
        
        return self

    def column_discretizer(self, column_name):
        """
        Given a column of the dataframe, discretize it. 

        :params column_name: str 
            Column in self.df.columns 

        :return None
        """
        # extract min and max
        min_, max_ = self.extrema.get(column_name).values()
        nb_class = self.nb_classes.get(column_name, 
                                       int(round(np.sqrt(self.df.shape[0]))))

        # used mapper
        step = (max_ - min_) / nb_class
        mapper_d = lambda x: x // step

        # discretize the column
        self.df[column_name] = self.df[column_name].apply(mapper_d)

    def column_indexer(self, column_name):
        """
        Given a column, index it ranging from 0 to nb of class desired 
        in the column (approximately).

        :params column_name: str
            Column in self.df.columns
        
        :return None
        """
        # discretize the column first
        self.column_discretizer(column_name)

        # get values 
        values = sorted(list(pd.unique(self.df[column_name])))

        # from discretization to index
        indexer = {}    
        reverse_indexer = {}
        for index, value in enumerate(values):
            indexer[value] = index
            reverse_indexer[index] = value

        # transform dataframe
        self.df[column_name] = self.df[column_name].apply(lambda x: indexer.get(x))

        # store transformation and reverse transformation definitely. 
        self._indexer[column_name] = indexer
        self._reverse_indexer[column_name] = reverse_indexer

    def indexer(self, column_name, x):
        """
        Function that takes as input a column name and a value of this 
        variable. Given that, it returns the index obtained during the fit. 
        If the discreted value is not seen yet, return None.

        :params column_name: str
            Column in the original dataframe. 
        :params x: float
            Value of corresponding to the column 

        :return int
            Index already encountered, None if  discretized value not seen yet.  
        """
        # extract min and max
        min_, max_ = self.extrema.get(column_name).values()
        nb_class = self.nb_classes.get(column_name, 
                                       int(round(np.sqrt(self.df.shape[0]))))

        # used mapper for discretization
        step = (max_ - min_) / nb_class
        x_discretized = x // step

        # get corresponding index
        x_index = self._indexer[column_name].get(x_discretized, None)

        return x_index

    def reverse_indexer(self, column_name, ind):
        """
        Given an index, return an approximation of its initial value. 
        """
        # extract min and max
        min_, max_ = self.extrema.get(column_name).values()
        nb_class = self.nb_classes.get(column_name, 
                                       int(round(np.sqrt(self.df.shape[0]))))

        # get step used for discretization
        step = (max_ - min_) / nb_class

        x_discretized = self._reverse_indexer[column_name].get(ind) * step

        return x_discretized


def pgmpy_friendly_transformer(df, sliding_window):
    """
    Given a pandas dataframe, reshape/transform it into a usable dataframe for 
    Dynamic bayesian training. 

    :params df: pandas.DataFrame
        Dataframe to transform. Each column represent a dynamic variable 
        whereas index are time steps. 
    :params sliding_window: int 
        Nb of successive time step to consider into a window. 

    :return pd.DataFrame.
    """
    # extract useful info from dataframe
    nb_points, nb_var = df.shape

    # convert dataframe into dictionary 
    df_dico = df.to_dict()

    # construct dataframe as required by DBN class
    matrix = np.zeros((nb_points-sliding_window-1, nb_var*sliding_window)) 

    for i in range(nb_points-sliding_window-1):
        for num_var, var in enumerate(df.columns):
            for w in range(sliding_window):
                matrix[i, w * nb_var + num_var] = df_dico[var][i+w]

    # name columns as required by DBN class
    columns = []
    for j in range(sliding_window):  
        for var in df.columns:
            columns.extend([(var, j)])

    # convert back to dataframe object
    new_df = pd.DataFrame(matrix, columns=columns).astype(int)

    return new_df


def test0():
    pass

if __name__ == "__main__":
    test0()
