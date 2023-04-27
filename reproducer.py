import sys
import argparse
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
  
  if(prim_kind == 'all' or temp[0] == prim_kind):
    if(float(temp[3]) > 0.0):       
      operations.update({'operation': temp[1]})
      operations.update({'primitive': temp[0]})
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
            
            
        curr_op = {'primitive':item['primitive'], 'operation':item['operation'],'ncalls':item['ncalls'], 'log1_time':a_time, 'log2_time':b_time,'delta':delta,'diff':diff}
        matches.append(curr_op)
      
      sorted_ops = sorted(matches, key=itemgetter('diff'))
  return sorted_ops
  

def generate_benchdnn_inputs(prim_kind, p_ops, p_ops_ncalls):
  prim_type = convert_driver(inputs.primitive_kind)
      
  benchdnn_input = generate_benchdnn_input(inputs.log2)
  benchdnn_file_name = 'benchdnn_inputs.' + str(prim_type)
  benchdnn_file = open(benchdnn_file_name, 'w')
      
  lines = benchdnn_input[1][prim_type]
  plines = lines.split("\n")
      
  for line in plines:
    temp_line = line.split(" ")
    for i in range(len(p_ops)):
        if p_ops[i] in temp_line[len(temp_line) -1]:
          fixed_time = "--fix-times-per-prb=" + str(p_ops_ncalls[i])
          temp_line.insert(2, fixed_time)
          curr_line = " ".join(str(x) for x in temp_line)
          curr_line = curr_line + "\n"
          benchdnn_file.write(curr_line)
          pass
            
  benchdnn_file.close()
    




def parse_args():
    parser=argparse.ArgumentParser(description=" ")
    parser.add_argument("-t", "--threshold", default="9999999")
    parser.add_argument("-m", "--max", default="0")
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
    

    a = prepare_list(log_breakdown1, inputs.primitive_kind);
    b = prepare_list(log_breakdown2, inputs.primitive_kind)
    
    sorted_ops = match_logs(a, b)
    
    problematic_ops = []
    problematic_ops_calls = []
    
    print("Primitive: " + " | Shape: " + " | Log1 time(ms): "  + " | Log2 time(ms): "  + " | Delta %: "  + " | Difference: " )
    
    counter = 0
    for i in sorted_ops:
        if((i['delta'] <= (-1 *(float(inputs.threshold))) ) and counter <= int(inputs.max)):
            problematic_ops.append(i['operation'])
            problematic_ops_calls.append(int(i['ncalls']))
            counter = counter +1
          
        print(str(i['primitive']) + " | " + i['operation'] + " | " + str(i['log1_time']) + " | " + str(i['log2_time']) + " | " + str(i['delta']) + " | " + str(i['diff']))
      
      
    print("Total matches: " + str(len(sorted_ops)) + " out of " + str(len(a)))
    
    if(inputs.generate):
      generate_benchdnn_inputs(inputs.primitive_kind, problematic_ops, problematic_ops_calls)

if __name__ == '__main__':
    main()
