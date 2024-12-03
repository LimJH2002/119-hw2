"""
Part 3: Measuring Performance

Now that you have drawn the dataflow graph in part 2,
this part will explore how performance of real pipelines
can differ from the theoretical model of dataflow graphs.

We will measure the performance of your pipeline
using your ThroughputHelper and LatencyHelper from HW1.

=== Coding part 1: making the input size and partitioning configurable ===

We would like to measure the throughput and latency of your PART1_PIPELINE,
but first, we need to make it configurable in:
(i) the input size
(ii) the level of parallelism.

Currently, your pipeline should have two inputs, load_input() and load_input_bigger().
You will need to change part1 by making the following additions:

- Make load_input and load_input_bigger take arguments that can be None, like this:

    def load_input(N=None, P=None)

    def load_input_bigger(N=None, P=None)

You will also need to do the same thing to q8_a and q8_b:

    def q8_a(N=None, P=None)

    def q8_b(N=None, P=None)

Here, the argument N = None is an optional parameter that, if specified, gives the size of the input
to be considered, and P = None is an optional parameter that, if specifed, gives the level of parallelism
(number of partitions) in the RDD.

You will need to make both functions work with the new signatures.
Be careful to check that the above changes should preserve the existing functionality of part1
(so python3 part1.py should still give the same output as before!)

Don't make any other changes to the function sigatures.

Once this is done, define a *new* version of the PART_1_PIPELINE, below,
that takes as input the parameters N and P.
(This time, you don't have to consider the None case.)
You should not modify the existing PART_1_PIPELINE.
"""

import os
import time

import matplotlib.pyplot as plt
import pyspark
from pyspark.sql import SparkSession

import part1

spark = SparkSession.builder.appName("PerformanceMeasurement").getOrCreate()
sc = spark.sparkContext

# Share the SparkSession with part1
import part1

part1.spark = spark
part1.sc = sc


def PART_1_PIPELINE_PARAMETRIC(N, P):
    """
    TODO: Follow the same logic as PART_1_PIPELINE
    N = number of inputs
    P = parallelism (number of partitions)
    (You can copy the code here), but make the following changes:
    - load_input should use an input of size N.
    - load_input_bigger (including q8_a and q8_b) should use an input of size N.
    - both of these should return an RDD with level of parallelism P (number of partitions = P).
    """

    def run_pipeline(rdd):
        results = []
        results.append(("q4", part1.q4(rdd)))
        results.append(("q5", part1.q5(rdd)))
        results.append(("q6", part1.q6(rdd)))
        results.append(("q7", part1.q7(rdd)))
        return results

    rdd = part1.load_input(N, P)

    results1 = run_pipeline(rdd)
    results2 = [("q8a", part1.q8_a(N, P)), ("q8b", part1.q8_b(N, P))]

    return results1 + results2


"""
=== Coding part 2: measuring the throughput and latency ===

Now we are ready to measure the throughput and latency.

To start, copy the code for ThroughputHelper and LatencyHelper from HW1 into this file.

Then, please measure the performance of PART1_PIPELINE as a whole
using five levels of parallelism:
- parallelism 1
- parallelism 2
- parallelism 4
- parallelism 8
- parallelism 16

For each level of parallelism, you should measure the throughput and latency as the number of input
items increases, using the following input sizes:
- N = 1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000

You can generate any plots you like (for example, a bar chart or an x-y plot on a log scale,)
but store them in the following 10 files,
where the file name corresponds to the level of parallelism:

output/part3-throughput-1.png
output/part3-throughput-2.png
output/part3-throughput-4.png
output/part3-throughput-8.png
output/part3-throughput-16.png
output/part3-latency-1.png
output/part3-latency-2.png
output/part3-latency-4.png
output/part3-latency-8.png
output/part3-latency-16.png

Clarifying notes:

- To control the level of parallelism, use the N, P parameters in your PART_1_PIPELINE_PARAMETRIC above.

- Make sure you sanity check the output to see if it matches what you would expect! The pipeline should run slower
  for larger input sizes N (in general) and for fewer number of partitions P (in general).

- For throughput, the "number of input items" should be 2 * N -- that is, N input items for load_input, and N for load_input_bigger.

- For latency, please measure the performance of the code on the entire input dataset
(rather than a single input row as we did on HW1).
MapReduce is batch processing, so all input rows are processed as a batch
and the latency of any individual input row is the same as the latency of the entire dataset.
That is why we are assuming the latency will just be the running time of the entire dataset.
"""


# Copy in ThroughputHelper and LatencyHelper
NUM_RUNS = 1  # Number of runs to average throughput and latency


class ThroughputHelper:
    def __init__(self):
        # Initialize the object.
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline input sizes
        self.sizes = []

        # Pipeline throughputs
        # This is set to None, but will be set to a list after throughputs
        # are calculated.
        self.throughputs = None

    def add_pipeline(self, name, size, func):
        self.pipelines.append(func)
        self.names.append(name)
        self.sizes.append(size)

    def compare_throughput(self):
        # Measure the throughput of all pipelines
        # and store it in a list in self.throughputs.
        # Make sure to use the NUM_RUNS variable.
        # Also, return the resulting list of throughputs,
        # in **number of items per second.**
        self.throughputs = []
        for func, size in zip(self.pipelines, self.sizes):
            start_time = time.time()
            for _ in range(NUM_RUNS):
                func()
            end_time = time.time()
            total_time = end_time - start_time
            items_per_second = (size * NUM_RUNS) / total_time
            self.throughputs.append(items_per_second)
        return self.throughputs


class LatencyHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline latencies
        # This is set to None, but will be set to a list after latencies
        # are calculated.
        self.latencies = None

    def add_pipeline(self, name, func):
        self.pipelines.append(func)
        self.names.append(name)

    def compare_latency(self):
        # Measure the latency of all pipelines
        # and store it in a list in self.latencies.
        # Also, return the resulting list of latencies,
        # in **milliseconds.**
        self.latencies = []
        for func in self.pipelines:
            total_time = 0
            for _ in range(NUM_RUNS):
                start_time = time.time()
                func()
                end_time = time.time()
                total_time += end_time - start_time
            # Convert to milliseconds and get average
            avg_latency = (total_time / NUM_RUNS) * 1000
            self.latencies.append(avg_latency)
        return self.latencies


def measure_performance(N, P):
    """Measure throughput and latency for given input size and parallelism"""
    throughput_helper = ThroughputHelper()
    latency_helper = LatencyHelper()

    # Create a pipeline function that handles both inputs
    def pipeline_func():
        PART_1_PIPELINE_PARAMETRIC(N, P)

    # Add pipeline to both helpers
    pipeline_name = f"N={N},P={P}"
    # For throughput we need total items processed (N from each input)
    throughput_helper.add_pipeline(pipeline_name, 2 * N, pipeline_func)
    latency_helper.add_pipeline(pipeline_name, pipeline_func)

    # Measure performance
    throughput = throughput_helper.compare_throughput()[0]  # First (only) pipeline
    latency = latency_helper.compare_latency()[0]  # First (only) pipeline

    return throughput, latency


def generate_plots():
    """Generate throughput and latency plots for different parallelism levels"""
    input_sizes = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    parallelism_levels = [1, 2, 4, 8, 16]

    os.makedirs("output", exist_ok=True)

    # For each parallelism level
    for P in parallelism_levels:
        throughputs = []
        latencies = []

        # Test each input size
        for N in input_sizes:
            throughput, latency = measure_performance(N, P)
            throughputs.append(throughput)
            latencies.append(latency)

        # Generate throughput plot
        plt.figure(figsize=(10, 6))
        plt.semilogx(input_sizes, throughputs, "b-o")
        plt.grid(True)
        plt.xlabel("Input Size (N)")
        plt.ylabel("Throughput (items/sec)")
        plt.title(f"Throughput vs Input Size (Parallelism = {P})")
        plt.savefig(f"output/part3-throughput-{P}.png")
        plt.close()

        # Generate latency plot
        plt.figure(figsize=(10, 6))
        plt.semilogx(input_sizes, latencies, "r-o")
        plt.grid(True)
        plt.xlabel("Input Size (N)")
        plt.ylabel("Latency (milliseconds)")  # Changed to milliseconds
        plt.title(f"Latency vs Input Size (Parallelism = {P})")
        plt.savefig(f"output/part3-latency-{P}.png")
        plt.close()


"""
=== Reflection part ===

Once this is done, write a reflection and save it in
a text file, output/part3-reflection.txt.

I would like you to think about and answer the following questions:

1. What would we expect from the throughput and latency
of the pipeline, given only the dataflow graph?

Use the information we have seen in class. In particular,
how should throughput and latency change when we double the amount of parallelism?

Please ignore pipeline and task parallelism for this question.
The parallelism we care about here is data parallelism.

2. In practice, does the expectation from question 1
match the performance you see on the actual measurements? Why or why not?

State specific numbers! What was the expected throughput and what was the observed?
What was the expected latency and what was the observed?

3. Finally, use your answers to Q1-Q2 to form a conjecture
as to what differences there are (large or small) between
the theoretical model and the actual runtime.
Name some overheads that might be present in the pipeline
that are not accounted for by our theoretical model of
dataflow graphs that could affect performance.

You should list an explicit conjecture in your reflection, like this:

    Conjecture: I conjecture that ....

You may have different conjectures for different parallelism cases.
For example, for the parallelism=4 case vs the parallelism=16 case,
if you believe that different overheads are relevant for these different scenarios.

=== Grading notes ===

Don't forget to fill out the entrypoint below before submitting your code!
Running python3 part3.py should work and should re-generate all of your plots in output/.

In the reflection, please write at least a paragraph for each question. (5 sentences each)

Please include specific numbers in your reflection (particularly for Q2).

=== Entrypoint ===
"""

if __name__ == "__main__":
    print(
        "Complete part 3. Please use the main function below to generate your plots so that they are regenerated whenever the code is run:"
    )
    generate_plots()
