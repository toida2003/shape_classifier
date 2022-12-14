?	????@????@!????@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails????@1????l???I*8? "?@r0:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"B84.8 % of the total step time sampled is spent on 'Kernel Launch'.*noIҳ_??2U@Qma??i.@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	????l???????l???!????l???*      ??!       2      ??!       :	*8? "?@*8? "?@!*8? "?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qҳ_??2U@yma??i.@?"w
Ktraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterb[???`??!b[???`??0"u
Jtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput??!??@??!0x,????0"/
conv2d_1/Relu_FusedConv2DosU??˧?!s$????"u
Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter??f?Q??!D/瘡???0"w
Ktraining/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterh?M?ܞ?!(u??0??0"w
Ktraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter%?+?h??!?-?0???0"u
Jtraining/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput???ĂԜ?!??\A???0"u
Jtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputI?^= ??!ˏ?2E???0"f
3loss/dense_1_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits6????!,??<??"/
conv2d_2/Relu_FusedConv2D??????!?j+'???Q      Y@Y;?;?;@a;?;?R@"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
highB84.8 % of the total step time sampled is spent on 'Kernel Launch'.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 