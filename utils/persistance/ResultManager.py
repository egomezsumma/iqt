
import os
import json

class ResultManager(object):

    def __init__(self, base_folder, exp_name, job_id, data={}):
        self.base_folder = base_folder
        self.exp_name = exp_name
        self.job_id = job_id
        self.data = data
        self.data['job_id'] = job_id
        self.directory = self.base_folder + "/" + self.exp_name + '.' + str(self.job_id)

    def add_data(self, key, sub_data):
        self.data[key] = sub_data

    def start(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            
        if not os.path.exists(self.directory+ "/data.json"):
            with open(self.directory + "/data.json", 'w') as outfile:
                json.dump(self.data, outfile, sort_keys = True, indent = 4, ensure_ascii=False)


