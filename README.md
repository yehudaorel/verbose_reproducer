# Verbose log reproducer

The purpose of this tool is to compare two oneDNN logs by primitive type and shape. 
Verbose log reproducer is a script building on top of the verbose_converter script.

## Requirements
 - Python 3.7

## Usage
### Option 1: call from command line
``` sh
python3 reproducer.py [-h] LOG1 LOG2 [-p PRIMITIVE] [-t THRESHOLD]
                            [-g] [-o] [-m MAX]
```

### Arguments
  - `{-h,--help}` -- display help message and exit.
  - `{LOG1} STRING` -- First verbose log file.
  - `{LOG2} STRING` -- Second verbose log file.
  - `{-p,--primitive} STRING` -- type of primitive. Default is `all`.
  - `{-t,--threshold} FLOAT` -- set minimum performance regression % to output. Default is `0.0`.
  - `{-g,--generate}` -- if passed, benchdnn reproducer file will be genereated.
  - `{-o,--output}` -- if passed, shape breakdown csv file will be genereated.
  - `{-m,--max} INT` -- shows N top results. no limit by default.



