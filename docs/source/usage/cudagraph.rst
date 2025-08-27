.. _using-nccl-with-cuda-graphs:

***************************
Using NCCL with CUDA Graphs
***************************

Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs.

CUDA Graphs provide a way to define workflows as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a single CPU operation. More details about CUDA Graphs can be found in the `CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>`_.

NCCL's collective, P2P and group operations all support CUDA Graph captures. This support requires a minimum CUDA version of 11.3.

The following sample code shows how to capture computational kernels and NCCL operations in a CUDA Graph: ::

  cudaGraph_t graph;
  cudaStreamBeginCapture(stream);
  kernel_A<<< ..., stream >>>(...);
  kernel_B<<< ..., stream >>>(...);
  ncclAllreduce(..., stream);
  kernel_C<<< ..., stream >>>(...);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

Starting with NCCL 2.11, when NCCL communication is captured and the CollNet algorithm is used, NCCL allows for further performance improvement via user buffer registration. For details, please environment variable :ref:`NCCL_GRAPH_REGISTER`.

Having multiple outstanding NCCL operations that are any combination of graph-captured or non-captured is supported. There is a caveat that the mechanism NCCL uses internally to accomplish this has been seen to cause CUDA to deadlock when the graphs of multiple communicators are cudaGraphLaunch()'d from the same thread. To disable this mechansim see the environment variable :ref:`NCCL_GRAPH_MIXING_SUPPORT`.