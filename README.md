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
  - `{-m,--max} INT` -- shows N top results. no limit by default.
  - `{--impl} ` -- shows operation kernel implimentation in breakdown.
  
  
### Directory tree
``` sh
verbose_reproducer/
├── README.md
├── reproducer.py
├── shape_analysis.csv
├── log1.txt
├── log2.txt
└── verbose_converter
    │ 
    ├── README.md
    ├── src
    │   ├── benchdnn_generator.py
    │   ├── breakdown_generator.py
    │   ├── dnnl_parser.py
    │   ├── utils.py
    │   └── writer.py
    ├── tests   
    └── verbose_converter.py
```
  ## Example runs
  
  ### Default
  By default, the script outputs 
 > Note: by default the script only outputs operations with delta of less than 0 (i.e with perf regression), ordered by difference.
 
``` sh
 python3 reproducer.py log1.txt log2.txt                                           
```

``` sh
 ------------------------------------------------------------------------------------------------------------------------------------------------------
 Primitive                              Shape                                 NCalls    Log1 time(ms)   Log2 time(ms)      Delta     Difference(ms)
 ------------------------------------------------------------------------------------------------------------------------------------------------------
 conv    mb1_ic1oc64_ih1000038oh1000000kh39sh1dh0ph0_iw20ow1kw20sw1dw0pw0     2.0     6083.27         6942.4          -14.1228%     -859.13
 conv    mb1_ic1oc32_ih1000020oh1000000kh21sh1dh0ph0_iw25ow1kw25sw1dw0pw0     3.0     4599.35         5450.12         -18.4976%     -850.77
 conv    mb1_ic1oc32_ih1000022oh1000000kh23sh1dh0ph0_iw20ow1kw20sw1dw0pw0     2.0     2707.54         3133.11         -15.718%      -425.57
 conv    mb1_ic1oc64_ih840538oh840500kh39sh1dh0ph0_iw20ow1kw20sw1dw0pw0       1.0     2553.63         2914.93         -14.1485%     -361.3
 conv    mb1_ic1oc128_ih1000004oh1000000kh5sh1dh0ph0_iw25ow1kw25sw1dw0pw0     3.0     3033.98         3388.29         -11.6781%     -354.31
 .
 . 
 .
 pool    mb1ic1_ih1060oh106kh10sh10dh0ph0_iw25ow25kw1sw1dw0pw0                1.0     0.03            0.04            -33.3333%     -0.01
 pool    mb1ic1_ih1211oh122kh10sh10dh0ph4_iw20ow20kw1sw1dw0pw0                1.0     0.02            0.03            -50.0%        -0.01
 pool    mb1ic1_ih1180oh118kh10sh10dh0ph0_iw20ow20kw1sw1dw0pw0                1.0     0.02            0.03            -50.0%        -0.01
 pool    mb1ic1_ih1161oh117kh10sh10dh0ph4_iw20ow20kw1sw1dw0pw0                1.0     0.02            0.03            -50.0%        -0.01
 pool    mb1ic1_ih1174oh118kh10sh10dh0ph3_iw20ow20kw1sw1dw0pw0                1.0     0.02            0.03            -50.0%        -0.01
 pool    mb1ic1_ih1109oh111kh10sh10dh0ph0_iw20ow20kw1sw1dw0pw0                1.0     0.02            0.03            -50.0%        -0.01
 pool    mb1ic1_ih1061oh107kh10sh10dh0ph4_iw20ow20kw1sw1dw0pw0                1.0     0.02            0.03            -50.0%        -0.01
 reorder 32x1x21x25                                                           1.0     0.01            0.02            -100.0%       -0.01
 Total matches: 6425 out of 6425
 Total operations found with 0.0% perf regression or more: 4890
```
  
### Threshold and primitive type

In this example we specify which primitive to parse (convolution) and the minimum amount of regression to output (t < -20)

``` sh
 python3 reproducer.py log1.txt log2.txt --primitive convolution -t -20                                           
```

``` sh
------------------------------------------------------------------------------------------------------------------------------------------------------
Primitive                              Shape                                 NCalls    Log1 time(ms)   Log2 time(ms)      Delta     Difference(ms)
------------------------------------------------------------------------------------------------------------------------------------------------------
conv    mb1_ic1oc64_ih81784oh81746kh39sh1dh0ph0_iw20ow1kw20sw1dw0pw0         1.0     248.07          353.77          -42.6089%     -105.7
conv    mb1_ic1oc32_ih81768oh81746kh23sh1dh0ph0_iw20ow1kw20sw1dw0pw0         1.0     110.73          182.77          -65.0592%     -72.04
conv    mb1_ic1oc32_ih9894oh9874kh21sh1dh0ph0_iw25ow1kw25sw1dw0pw0           13.0    198.65          243.1           -22.376%      -44.45
conv    mb1_ic1oc32_ih122oh100kh23sh1dh0ph0_iw20ow1kw20sw1dw0pw0            1101.0   197.57          238.43          -20.6813%     -40.86
conv    mb1_ic1oc32_ih24451oh24431kh21sh1dh0ph0_iw25ow1kw25sw1dw0pw0         2.0     75.11           111.28          -48.156%      -36.17
conv    mb1_ic1oc32_ih24448oh24428kh21sh1dh0ph0_iw25ow1kw25sw1dw0pw0         2.0     83.53           110.12          -31.8329%     -26.59
conv    mb1_ic1oc32_ih13651oh13631kh21sh1dh0ph0_iw25ow1kw25sw1dw0pw0         3.0     62.67           86.68           -38.3118%     -24.01
conv    mb1_ic1oc128_ih24435oh24431kh5sh1dh0ph0_iw25ow1kw25sw1dw0pw0         2.0     49.67           71.28           -43.5071%     -21.61
 .
 . 
 .
conv    mb1_ic1oc128_ih176oh172kh5sh1dh0ph0_iw25ow1kw25sw1dw0pw0             1.0     0.16            0.21            -31.25%       -0.05
conv    mb1_ic1oc64_ih162oh154kh9sh1dh0ph0_iw25ow1kw25sw1dw0pw0              1.0     0.14            0.19            -35.7143%     -0.05
conv    mb1_ic1oc64_ih150oh142kh9sh1dh0ph0_iw25ow1kw25sw1dw0pw0              1.0     0.14            0.19            -35.7143%     -0.05
conv    mb1_ic1oc64_ih147oh139kh9sh1dh0ph0_iw25ow1kw25sw1dw0pw0              1.0     0.12            0.17            -41.6667%     -0.05
conv    mb1_ic1oc32_ih139oh119kh21sh1dh0ph0_iw25ow1kw25sw1dw0pw0             1.0     0.19            0.23            -21.0526%     -0.04
conv    mb1_ic1oc64_ih127oh119kh9sh1dh0ph0_iw25ow1kw25sw1dw0pw0              1.0     0.11            0.14            -27.2727%     -0.03
conv    mb1_ic1oc64_ih111oh103kh9sh1dh0ph0_iw25ow1kw25sw1dw0pw0              1.0     0.09            0.11            -22.2222%     -0.02
Total matches: 4908 out of 4908
Total operations found with -20% perf regression or more: 679
```
You can also limit the amount of operations by setting the paramater --max or -m:

``` sh
 python3 reproducer.py log1.txt log2.txt --primitive convolution -t -20 -m 3                                          
```

``` sh
 ------------------------------------------------------------------------------------------------------------------------------------------------------
 Primitive                              Shape                                 NCalls    Log1 time(ms)   Log2 time(ms)      Delta     Difference(ms)
 ------------------------------------------------------------------------------------------------------------------------------------------------------
 conv    mb1_ic1oc64_ih81784oh81746kh39sh1dh0ph0_iw20ow1kw20sw1dw0pw0         1.0     248.07          353.77          -42.6089%     -105.7
 conv    mb1_ic1oc32_ih81768oh81746kh23sh1dh0ph0_iw20ow1kw20sw1dw0pw0         1.0     110.73          182.77          -65.0592%     -72.04
 conv    mb1_ic1oc32_ih9894oh9874kh21sh1dh0ph0_iw25ow1kw25sw1dw0pw0           13.0    198.65          243.1           -22.376%      -44.45
 Total matches: 4908 out of 4908
 Total operations found with -20% perf regression or more: 3
```

