from datetime import datetime, timedelta

class Logger:
    def __init__(self):
        self.time_log = []
        self.val_scores = []
        self.train_loss = []
        self.other_loss = []
        self.current_time = timedelta(0)
        self.current_stage = -1
    
    def start_new_stage(self):
        self.time_log.append([])
        self.val_scores.append([])
        self.train_loss.append([])
        self.other_loss.append([])
        if self.current_stage != -1:
            self.current_time = self.time_log[self.current_stage][-1]
        self.current_stage += 1

    def record(self, time_elapsed, val_score, train_loss, distill_knowledge=False, other_loss=0):
        current_stage = self.current_stage

        self.time_log[current_stage] += [self.current_time + time_elapsed]
        self.val_scores[current_stage] += [val_score]
        self.train_loss[current_stage] += [train_loss.cpu().detach().numpy()]
        if distill_knowledge:
            self.other_loss[current_stage] += [other_loss.cpu().detach().numpy()]
        else:
            self.other_loss[current_stage] += [0]
    
    def return_logs(self):
        return self.time_log, self.val_scores, self.train_loss, self.other_loss
    

def stage_to_string(stage, stages_configurations):
    string_name = ''
    for k,v in stages_configurations[stage].items():
        if k == 'repeated' and v > 0:
            string_name = '_'.join([string_name, 'extended-{}'.format(v)])
            continue
        if k == 'repeated' and v == 0:
            continue
        dimension = k.split('_')[0]
        stage_number = len(v)
        string_name = '_'.join([string_name, '{}-s{}'.format(dimension, stage_number)]) if string_name != '' else '{}-s{}'.format(dimension, stage_number)
        
    return string_name