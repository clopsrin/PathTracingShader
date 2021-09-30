# PathTracingShader
A naive path tracing shader based on Monte Carlo Method and Lambertian model.

Modify following arguments on the top of main.cpp file to adjust quality and performance.
+ GSampleTimes
+ GTracingDepth
+ GProcessNum

## Preview
5 depth, 40 spp (24s)
![](https://github.com/clopsrin/PathTracingShader/blob/main/images/40spp.png)

5 depth, 200 spp (122s)
![](https://github.com/clopsrin/PathTracingShader/blob/main/images/200spp.png)

5 depth, 2000 spp (1200s)
![](https://github.com/clopsrin/PathTracingShader/blob/main/images/2000spp.png)

5 depth, 5000 spp (3100s)
![](https://github.com/clopsrin/PathTracingShader/blob/main/images/5000spp.png)
