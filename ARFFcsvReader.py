"""
    Author: Shameer Sathar
"""

import csv


class ARFFcsvReader:
    """
    Operates on WEKA generated output file of predictions.
    The file is parsed and the predictions are returned as 1D numpy array.
    """
    def __init__(self, file_name):
        """
        :param file_name: WEKA generated csv file name including full path.
        :return: predicted values as list
        """
        self.file_name = file_name
        # read the csv file
        f_obj = open(self.file_name, "rb")
        reader = csv.reader(f_obj, delimiter=',')
        # Skip the header
        reader.next()
        self.prediction = []
        for line in self.skip_last(reader):
            # save the instance number and predictions given as
            # 9,1:?,2:0.0,,1 >
            vals = line[2].split(":")[1]
            self.prediction.append(int(float(vals)))
        print len(self.prediction)

    def skip_last(self, iterator):
        """
        To help skipping the last line while reading ARFF output file.
        :param iterator: File reader object
        :return: NULL
        """
        prev = next(iterator)
        for item in iterator:
            yield prev
            prev = item

    def get_prediction(self):
        """
        Get function to return the array
        :return: the predicted values as an array
        """
        return self.prediction
