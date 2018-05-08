class City(object):

    def __init__(self, city):

        if city == 'Mel':
            self.size = 3619  # 3619
            self.test_size = 3619
            self.learning_rate = 0.001
            self.d = 500  # 500
<<<<<<< HEAD
            self.epoch = 10
=======
            self.epoch = 1
>>>>>>> 8356e0be015e409127ec4d1d7f4b12edb5d1a711
            self.unit = 100
            self.batch_size = 3619
            self.location = 'Melbourne'

        if city == 'NY':
            self.size = 8105
            self.test_size = 8105
            self.learning_rate = 0.001
            self.d = 1000
            self.epoch = 100
            self.unit = 100
            self.batch_size = 8105
            self.location = 'NewYork'

        if city == 'SL':
            self.size = 36545
            self.test_size = 36545
            self.learning_rate = 0.001
            self.d = 5000
            self.epoch = 5
            self.unit = 100
            self.batch_size = 5000
            self.location = 'smallLondon'

    def name(self, time):
        filename = self.location + "_NN" + time + "_" + \
            str(self.unit) + "Units" + str(self.epoch) + \
            "Epochs" + str(self.learning_rate) + "Rate"

        return filename
