import sys
import argparse
import csv
import time
from operator import itemgetter

sys.path.insert(1, './verbose_converter/')
import verbose_converter


def convert_driver(prop_kind):
    driver = {
        'batch_normalization': 'bnorm',
        'binary': 'binary',
        'concat': 'concat',
        'convolution': 'conv',
        'deconvolution': 'deconv',
        'eltwise': 'eltwise',
        'inner_product': 'ip',
        'layer_normalization': 'lnorm',
        'layer_normalization_v2': 'lnorm',
        'lrn': 'lrn',
        'matmul': 'matmul',
        'pooling': 'pool',
        'pooling_v2': 'pool',
        'prelu': 'prelu',
        'reduction': 'reduction',
        'reorder': 'reorder',
        'resampling': 'resampling',
        'rnn': 'rnn',
        'shuffle': 'shuffle',
        'softmax': 'softmax',
        'softmax_v2': 'softmax',
        'sum': 'sum',
        'all': 'all',
    }.get(prop_kind)
    return driver
    
def cleanup(breakdown):
    temp = ' '.join([str(elem) for i,elem in enumerate(breakdown)])
    parsed_breakdown = temp.split("'")[3]
    return parsed_breakdown.split("\\n")



def parse_log(log):
  cur_log = open(log, "r")
  output = verbose_converter.convert(0, 'oneDNN', cur_log, 'generate', 'breakdown', 1, agg_keys=['prim_kind', 'shapes'])
    
  log_breakdown = cleanup(output)
  cur_log.close()
  return log_breakdown



def generate_benchdnn_input(log):
  cur_log = open(log, "r")
  benchdnn_input = verbose_converter.convert(0, 'oneDNN', cur_log, 'generate', 'benchdnn', 1, agg_keys=['prim_kind'])
    
  cur_log.close()
  return benchdnn_input



def prepare_map(breakdown, prim_kind):
  operations = {}
  temp = breakdown.split(',')
  
  if((prim_kind == 'all' or temp[0] == prim_kind) and (float(temp[3]) > 0.0)):       
      operations.update({'operation': temp[1]})
      operations.update({'primitive': convert_driver(temp[0])})
      operations.update({'ncalls': float(temp[2])})
      operations.update({'time': float(temp[3])})
  
  return operations


  
def prepare_list(breakdown, prim_kind = 'all'):
  map_list = []
  # Extract number of calls, exec time and kind from breakdown
  if len(breakdown) > 1:
    for i in range(1, len(breakdown)):
      current = prepare_map(breakdown[i], prim_kind)
      if(current):
        map_list.append(current)
  else:
    print("Log breakdown is empty!")

  return map_list



def match_logs(a, b):
  matches = []
    
  b_dict = {x['operation']: x for x in b}

  for item in a:
    if item['operation'] in b_dict:
      key = item['operation']
      if(item['ncalls'] == b_dict[key]['ncalls']):
        a_time = float(item['time'])
        b_time = float(b_dict[key]['time'])
            
        delta = (((a_time-b_time)/a_time) * 100)
        diff = (a_time-b_time)
            
            
        curr_op = {'primitive':item['primitive'],
                   'operation':item['operation'],
                   'ncalls':item['ncalls'],
                   'log1_time':a_time,
                   'log2_time':b_time,
                   'delta':delta,
                   'diff':diff}

        matches.append(curr_op)
      
      sorted_ops = sorted(matches, key=itemgetter('diff'))

  return sorted_ops
  

def output_benchdnn_inputs(lines, prim):
  benchdnn_file_name = 'benchdnn_inputs.' + str(prim)
  with open(benchdnn_file_name, 'w', encoding='UTF8') as output_file:
    for line in lines:
      output_file.write(line)
      
     


def generate_benchdnn_inputs(prim_kind, benchdnn_input, p_ops):
  
  if(prim_kind == "all"):
    p_ops_prims = [str(elem['primitive']) for i,elem in enumerate(p_ops)]
    prim_intersection = list(set(benchdnn_input[1].keys()) & set(p_ops_prims))
    prim_keys = benchdnn_input[1].keys()
  else:
    prim_keys = [prim_kind]
  

  for key in prim_keys:
    
    lines = benchdnn_input[1][key]
    plines = lines.split("\n")
    input_lines = []
    
    for line in plines:
      temp_line = line.split(" ")
      for y, i in enumerate(p_ops):
        if str(i['operation']) == temp_line[len(temp_line) -1]:
          fixed_time = "--fix-times-per-prb=" + str(i['ncalls'])
          temp_line.insert(2, fixed_time)
          curr_line = " ".join(str(x) for x in temp_line)
          curr_line = curr_line + "\n"
          
          input_lines.append(curr_line)
    
          del p_ops[y]
          pass
          
    if(len(input_lines) > 0):
      output_benchdnn_inputs(input_lines, key)
  
  

def print_shape_analysis(sorted_ops, is_output):
  header = ["Primitive" , "Shape" , "NCalls" , "Log1 time(ms)" , "Log2 time(ms)" , "Delta %" , "Difference"]
  
  with open('shape_analysis.csv', 'w', encoding='UTF8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    
    




def parse_args():
    parser=argparse.ArgumentParser(description=" ")
    parser.add_argument("-t", "--threshold", default="0.0")
    parser.add_argument("-m", "--max")
    parser.add_argument("-p", "--primitive_kind", default="all")
    parser.add_argument("log1")
    parser.add_argument("log2")
    
    parser.add_argument('-g', '--generate',
                    action='store_true')
    parser.add_argument('-o', '--output',
                    action='store_true')
                    
                    
    args=parser.parse_args()
    return args



def main():
    inputs=parse_args()
    
    log_breakdown1 = parse_log(inputs.log1)
    log_breakdown2 = parse_log(inputs.log2)
    
    print(convert_driver("conv"))
    

    a = prepare_list(log_breakdown1, inputs.primitive_kind);
    b = prepare_list(log_breakdown2, inputs.primitive_kind)
    
    sorted_ops = match_logs(a, b)
    
    ops = []
    
    
    counter = 0
    for i in sorted_ops:
        if((i['delta'] < (1 *(float(inputs.threshold))))):
            counter = counter +1
            ops.append(i)
            print(str(i['primitive']) + " | " + i['operation'] + " | " + str(i['log1_time']) + " | " + str(i['log2_time']) + " | " + str(i['delta']) + " | " + str(i['diff']))
      
      
    print("Total matches: " + str(len(sorted_ops)) + " out of " + str(len(a)))
    
    if(inputs.generate):
      benchdnn_input = generate_benchdnn_input(inputs.log2)
      prim = convert_driver(inputs.primitive_kind)
      if(prim != None):
        generate_benchdnn_inputs(prim, benchdnn_input, ops)
        
      



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

