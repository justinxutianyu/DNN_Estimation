class City(object):
    
    def __init__(self, city):
        
        if city == 'Mel':
            self.size = 3619  # 3619 
            self.test_size = 3619 
            self.learning_rate = 0.001
            self.d = 3619  # 500
            self.epoch = 1
            self.unit = 100
            self.batch_size = 3619
            self.location = 'Melbourne'

        if city == 'NY':
            self.size = 8105
            self.test_Size = 8105
            self.learning_rate = 0.001
            self.d = 1000
            self.epoch = 20
            self.unit = 100
            self.batch_size = 8105
            self.location = 'NewYork'

    def name(self, time):
        filename = self.location+"_NN" + time+"_"+str(self.Units) + "Units" + str(self.epochs) + "Epochs" + str(self.learning_rate) + "Rate"

        return filename