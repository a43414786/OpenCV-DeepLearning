import os
import pickle

class record:
    def __init__(self):
        self.data = 0
        self.load()
    def add(self,info):
        self.data.append(info)
    def load(self):
        dirs = os.listdir()
        if "record.pickle" in dirs:
            with open("record.pickle","rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = [] 
    def dump(self):
        with open("record.pickle","wb") as f:
            pickle.dump(self.data,f)
    def pr_info(self):
        for i in self.data:
            print(i)
 