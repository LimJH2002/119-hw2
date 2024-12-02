"""
Part 1: MapReduce

In our first part, we will practice using MapReduce
to create several pipelines.
This part has 20 questions.

As you complete your code, you can run the code with

    python3 part1.py
    pytest part1.py

and you can view the output so far in:

    output/part1-answers.txt

In general, follow the same guidelines as in HW1!
Make sure that the output in part1-answers.txt looks correct.
See "Grading notes" here:
https://github.com/DavisPL-Teaching/119-hw1/blob/main/part1.py

For Q5-Q7, make sure your answer uses general_map and general_reduce as much as possible.
You will still need a single .map call at the beginning (to convert the RDD into key, value pairs), but after that point, you should only use general_map and general_reduce.

If you aren't sure of the type of the output, please post a question on Piazza.
"""

# Spark boilerplate (remember to always add this at the top of any Spark file)
import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DataflowGraphExample").getOrCreate()
sc = spark.sparkContext

# Additional imports
import pytest

"""
===== Questions 1-3: Generalized Map and Reduce =====

We will first implement the generalized version of MapReduce.
It works on (key, value) pairs:

- During the map stage, for each (key1, value1) pairs we
  create a list of (key2, value2) pairs.
  All of the values are output as the result of the map stage.

- During the reduce stage, we will apply a reduce_by_key
  function (value2, value2) -> value2
  that describes how to combine two values.
  The values (key2, value2) will be grouped
  by key, then values of each key key2
  will be combined (in some order) until there
  are no values of that key left. It should end up with a single
  (key2, value2) pair for each key.

1. Fill in the general_map function
using operations on RDDs.

If you have done it correctly, the following test should pass.
(pytest part1.py)

Don't change the q1() answer. It should fill out automatically.
"""


def general_map(rdd, f):
    """
    rdd: an RDD with values of type (k1, v1)
    f: a function (k1, v1) -> List[(k2, v2)]
    output: an RDD with values of type (k2, v2)
    """
    return rdd.flatMap(lambda pair: f(pair[0], pair[1]))


def test_general_map():
    rdd = sc.parallelize(["cat", "dog", "cow", "zebra"])

    # Use first character as key
    rdd1 = rdd.map(lambda x: (x[0], x))

    # Map returning no values
    rdd2 = general_map(rdd1, lambda k, v: [])

    # Map returning length
    rdd3 = general_map(rdd1, lambda k, v: [(k, len(v))])
    rdd4 = rdd3.map(lambda pair: pair[1])

    # Map returnning odd or even length
    rdd5 = general_map(rdd1, lambda k, v: [(len(v) % 2, ())])

    assert rdd2.collect() == []
    assert sum(rdd4.collect()) == 14
    assert set(rdd5.collect()) == set([(1, ())])


def q1():
    # Answer to this part: don't change this
    rdd = sc.parallelize(["cat", "dog", "cow", "zebra"])
    rdd1 = rdd.map(lambda x: (x[0], x))
    rdd2 = general_map(rdd1, lambda k, v: [(1, v[-1])])
    return sorted(rdd2.collect())


"""
2. Fill in the reduce function using operations on RDDs.

If you have done it correctly, the following test should pass.
(pytest part1.py)

Don't change the q2() answer. It should fill out automatically.
"""


def general_reduce(rdd, f):
    """
    rdd: an RDD with values of type (k2, v2)
    f: a function (v2, v2) -> v2
    output: an RDD with values of type (k2, v2),
        and just one single value per key
    """
    return rdd.reduceByKey(f)


def test_general_reduce():
    rdd = sc.parallelize(["cat", "dog", "cow", "zebra"])

    # Use first character as key
    rdd1 = rdd.map(lambda x: (x[0], x))

    # Reduce, concatenating strings of the same key
    rdd2 = general_reduce(rdd1, lambda x, y: x + y)
    res2 = set(rdd2.collect())

    # Reduce, adding lengths
    rdd3 = general_map(rdd1, lambda k, v: [(k, len(v))])
    rdd4 = general_reduce(rdd3, lambda x, y: x + y)
    res4 = sorted(rdd4.collect())

    assert res2 == set([("c", "catcow"), ("d", "dog"), ("z", "zebra")]) or res2 == set(
        [("c", "cowcat"), ("d", "dog"), ("z", "zebra")]
    )
    assert res4 == [("c", 6), ("d", 3), ("z", 5)]


def q2():
    # Answer to this part: don't change this
    rdd = sc.parallelize(["cat", "dog", "cow", "zebra"])
    rdd1 = rdd.map(lambda x: (x[0], x))
    rdd2 = general_reduce(rdd1, lambda x, y: "hello")
    return sorted(rdd2.collect())


"""
3. Name one scenario where having the keys for Map
and keys for Reduce be different might be useful.

=== ANSWER Q3 BELOW ===
One scenario where having different keys for Map and Reduce is useful is when processing log files. 
The Map stage might use timestamps as keys to group events, 
while the Reduce stage could use event types as keys to aggregate statistics across different time periods.
=== END OF Q3 ANSWER ===
"""

"""
===== Questions 4-10: MapReduce Pipelines =====

Now that we have our generalized MapReduce function,
let's do a few exercises.
For the first set of exercises, we will use a simple dataset that is the
set of integers between 1 and 1 million (inclusive).

4. First, we need a function that loads the input.
"""


def load_input():
    # Return a parallelized RDD with the integers between 1 and 1,000,000
    # This will be referred to in the following questions.
    return sc.parallelize(range(1, 1_000_001))


def q4(rdd):
    # Input: the RDD from load_input
    # Output: the length of the dataset.
    # You may use general_map or general_reduce here if you like (but you don't have to) to get the total count.
    return rdd.count()


"""
Now use the general_map and general_reduce functions to answer the following questions.

5. Among the numbers from 1 to 1 million, what is the average value?
"""


def q5(rdd):
    # Input: the RDD from Q4
    # Output: the average value
    mapped = general_map(rdd.map(lambda x: (1, x)), lambda k, v: [(1, (v, 1))])

    # Reduce to get sum and count
    reduced = general_reduce(mapped, lambda a, b: (a[0] + b[0], a[1] + b[1]))

    # Calculate average from the single (sum, count) pair
    result = reduced.map(lambda x: x[1][0] / x[1][1]).collect()[0]
    return result


"""
6. Among the numbers from 1 to 1 million, when written out,
which digit is most common, with what frequency?
And which is the least common, with what frequency?

(If there are ties, you may answer any of the tied digits.)

The digit should be either an integer 0-9 or a character '0'-'9'.
Frequency is the number of occurences of each value.

Your answer should use the general_map and general_reduce functions as much as possible.
"""


def q6(rdd):
    # Input: the RDD from Q4
    # Output: a tuple (most common digit, most common frequency, least common digit, least common frequency)
    def map_digits(k, v):
        return [(d, 1) for d in str(v)]

    # Map phase
    mapped = general_map(rdd.map(lambda x: (1, x)), map_digits)

    # Reduce phase to count frequencies
    counts = general_reduce(mapped, lambda a, b: a + b)

    # Convert to regular Python collection and find min/max
    freq_counts = counts.collect()
    max_digit = max(freq_counts, key=lambda x: x[1])
    min_digit = min(freq_counts, key=lambda x: x[1])

    return (max_digit[0], max_digit[1], min_digit[0], min_digit[1])


"""
7. Among the numbers from 1 to 1 million, written out in English, which letter is most common?
With what frequency?
The least common?
With what frequency?

(If there are ties, you may answer any of the tied characters.)

For this part, you will need a helper function that computes
the English name for a number.
Examples:

    0 = zero
    71 = seventy one
    513 = five hundred and thirteen
    801 = eight hundred and one
    999 = nine hundred and ninety nine
    500,501 = five hundred thousand five hundred and one
    555,555 = five hundred and fifty five thousand five hundred and fifty five
    1,000,000 = one million

Notes:
- Please ignore spaces and hyphens.
- Use lowercase letters.
- The word "and" should always appear after the "hundred" part (where present),
  but nowhere else.
- Please implement this without using an external library such as `inflect`.
"""


# *** Define helper function(s) here ***
def number_to_words(n):
    """Convert a number to its English representation"""
    ones = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    def process_chunk(num, index):
        if num == 0:
            return ""

        chunk_words = []

        # Handle hundreds
        if num >= 100:
            chunk_words.append(f"{ones[num // 100]} hundred")
            if num % 100 != 0:
                chunk_words.append("and")
            num %= 100

        # Handle tens and ones
        if num > 0:
            if num < 20:
                chunk_words.append(ones[num])
            else:
                tens_word = tens[num // 10]
                ones_word = ones[num % 10]
                if ones_word:
                    chunk_words.append(f"{tens_word} {ones_word}")
                else:
                    chunk_words.append(tens_word)

        # Add scale word if needed
        if index == 1 and chunk_words:  # Thousands
            chunk_words.append("thousand")
        elif index == 2 and chunk_words:  # Millions
            chunk_words.append("million")

        return " ".join(chunk_words)

    if n == 0:
        return "zero"

    # Split number into chunks of 3 digits
    chunks = []
    while n > 0:
        chunks.append(n % 1000)
        n //= 1000

    # Process each chunk
    words = []
    for i, chunk in enumerate(chunks):
        chunk_words = process_chunk(chunk, i)
        if chunk_words:
            words.append(chunk_words)

    return " ".join(reversed(words))


def q7(rdd):
    # Input: the RDD from Q4
    # Output: a tulpe (most common char, most common frequency, least common char, least common frequency)
    def map_letters(k, v):
        # Convert number to words and count each letter
        word = number_to_words(v).replace(" ", "").replace("-", "")
        return [(c, 1) for c in word if c.isalpha()]

    # Map each number to (letter, 1) pairs
    mapped = general_map(rdd.map(lambda x: (1, x)), map_letters)

    # Reduce to get letter counts
    counts = general_reduce(mapped, lambda a, b: a + b)

    # Find most and least common letters
    freq_counts = counts.collect()
    max_letter = max(freq_counts, key=lambda x: x[1])
    min_letter = min(freq_counts, key=lambda x: x[1])

    return (max_letter[0], max_letter[1], min_letter[0], min_letter[1])


"""
8. Does the answer change if we have the numbers from 1 to 100,000,000?

Make a version of both pipelines from Q6 and Q7 for this case.
You will need a new load_input function.
"""


def load_input_bigger():
    return sc.parallelize(range(1, 100_000_001))


def q8_a():
    # version of Q6
    # It should call into q6() with the new RDD!
    # Don't re-implemented the q6 logic.
    # Output: a tuple (most common digit, most common frequency, least common digit, least common frequency)
    return q6(load_input_bigger())


def q8_b():
    # version of Q7
    # It should call into q7() with the new RDD!
    # Don't re-implemented the q6 logic.
    # Output: a tulpe (most common char, most common frequency, least common char, least common frequency)
    return q7(load_input_bigger())


"""
Discussion questions

9. State the values of k1, v1, k2, and v2 for your Q6 and Q7 pipelines.

=== ANSWER Q9 BELOW ===
Q6:
k1: Any value (use 1 for simplicity)
v1: The original number
k2: Individual digits ('0'-'9')
v2: Count of occurrences

Q7:
k1: Any value (use 1 for simplicity)
v1: The original number
k2: Individual letters ('a'-'z')
v2: Count of occurrences
=== END OF Q9 ANSWER ===

10. Do you think it would be possible to compute the above using only the
"simplified" MapReduce we saw in class? Why or why not?

=== ANSWER Q10 BELOW ===
The simplified MapReduce from class would be insufficient because it only allows mapping items to exactly one key-value pair. 
In Q6 and Q7, we need to emit multiple key-value pairs for each input (one for each digit/letter), 
which requires the more general form of MapReduce.
=== END OF Q10 ANSWER ===
"""

"""
===== Questions 11-18: MapReduce Edge Cases =====

For the remaining questions, we will explore two interesting edge cases in MapReduce.

11. One edge case occurs when there is no output for the reduce stage.
This can happen if the map stage returns an empty list (for all keys).

Demonstrate this edge case by creating a specific pipeline which uses
our data set from Q4. It should use the general_map and general_reduce functions.

Output a set of (key, value) pairs after the reduce stage.
"""


def q11(rdd):
    # Input: the RDD from Q4
    # Output: the result of the pipeline, a set of (key, value) pairs
    return set(
        general_reduce(
            general_map(
                rdd.map(lambda x: (1, x)), lambda k, v: []
            ),  # Empty list for all inputs
            lambda x, y: x,
        ).collect()
    )


"""
12. What happened? Explain below.
Does this depend on anything specific about how
we chose to define general_reduce?

=== ANSWER Q12 BELOW ===
The result of q11 is an empty set because the map stage returns an empty list for every input value. 
When the map phase produces no key-value pairs at all, there's nothing for the reduce phase to process. 
This behavior doesn't depend on how general_reduce is implemented specifically - if there are no values to reduce, 
there will be no output regardless of the reduction function. 
=== END OF Q12 ANSWER ===

13. Lastly, we will explore a second edge case, where the reduce stage can
output different values depending on the order of the input.
This leads to something called "nondeterminism", where the output of the
pipeline can even change between runs!

First, take a look at the definition of your general_reduce function.
Why do you imagine it could be the case that the output of the reduce stage
is different depending on the order of the input?

=== ANSWER Q13 BELOW ===
The reduce operation processes pairs of values using the reduction function, 
but the order in which these pairs are combined can vary. 
If the reduce function is not both commutative (a+b = b+a) and associative ((a+b)+c = a+(b+c)), 
then different orderings can produce different results. 
=== END OF Q13 ANSWER ===

14.
Now demonstrate this edge case concretely by writing a specific example below.
As before, you should use the same dataset from Q4.

Important: Please create an example where the output of the reduce stage is a set of (integer, integer) pairs.
(So k2 and v2 are both integers.)
"""


def q14(rdd):
    # Map to (key, value) pairs where key is just remainder mod 2
    # This ensures we have just 2 groups where order will matter
    mapped = general_map(rdd.map(lambda x: (1, x)), lambda k, v: [(v % 2, v)])

    # Simple non-commutative operation: subtract
    reduced = general_reduce(mapped, lambda x, y: x - y)

    return set(reduced.collect())


"""
15.
Run your pipeline. What happens?
Does it exhibit nondeterministic behavior on different runs?
(It may or may not! This depends on the Spark scheduler and implementation,
including partitioning.

=== ANSWER Q15 BELOW ===
Q14  exhibits nondeterministic behavior because the reducer function keeps the larger number between pairs but does this across partitions in an unpredictable order. 
Running it multiple times might give different results since Spark makes no guarantees about the order of processing within partitions. 
The exact behavior depends on how Spark schedules the tasks and how data gets distributed across partitions.
=== END OF Q15 ANSWER ===

16.
Lastly, try the same pipeline with at least 3 different levels of parallelism.

Write three functions, a, b, and c that use different levels of parallelism.
"""


def q16_a():
    # For this one, create the RDD yourself. Choose the number of partitions.
    # TODO
    raise NotImplementedError


def q16_b():
    # For this one, create the RDD yourself. Choose the number of partitions.
    # TODO
    raise NotImplementedError


def q16_c():
    # For this one, create the RDD yourself. Choose the number of partitions.
    # TODO
    raise NotImplementedError


"""
Discussion questions

17. Was the answer different for the different levels of parallelism?

=== ANSWER Q17 BELOW ===

=== END OF Q17 ANSWER ===

18. Do you think this would be a serious problem if this occured on a real-world pipeline?
Explain why or why not.

=== ANSWER Q18 BELOW ===

=== END OF Q18 ANSWER ===

===== Q19-20: Further reading =====

19.
The following is a very nice paper
which explores this in more detail in the context of real-world MapReduce jobs.
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/icsecomp14seip-seipid15-p.pdf

Take a look at the paper. What is one sentence you found interesting?

=== ANSWER Q19 BELOW ===

=== END OF Q19 ANSWER ===

20.
Take one example from the paper, and try to implement it using our
general_map and general_reduce functions.
For this part, just return the answer "True" at the end if you found
it possible to implement the example, and "False" if it was not.
"""


def q20():
    # TODO
    raise NotImplementedError


"""
That's it for Part 1!

===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
everything together in a pipeline for you below.

Check out the output in output/part1-answers.txt.
"""

ANSWER_FILE = "output/part1-answers.txt"
UNFINISHED = 0


def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, "a") as f:
            f.write(f"{name},{answer}\n")
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, "a") as f:
            f.write(f"{name},Not Implemented\n")
        global UNFINISHED
        UNFINISHED += 1


def PART_1_PIPELINE():
    open(ANSWER_FILE, "w").close()

    try:
        dfs = load_input()
    except NotImplementedError:
        print("Welcome to Part 1! Implement load_input() to get started.")
        dfs = sc.parallelize([])

    # Questions 1-3
    log_answer("q1", q1)
    log_answer("q2", q2)
    # 3: commentary

    # Questions 4-10
    log_answer("q4", q4, dfs)
    log_answer("q5", q5, dfs)
    log_answer("q6", q6, dfs)
    log_answer("q7", q7, dfs)
    # log_answer("q8a", q8_a)
    # log_answer("q8b", q8_b)
    # 9: commentary
    # 10: commentary

    # Questions 11-18
    log_answer("q11", q11, dfs)
    # 12: commentary
    # 13: commentary
    log_answer("q14", q14, dfs)
    # 15: commentary
    log_answer("q16a", q16_a)
    log_answer("q16b", q16_b)
    log_answer("q16c", q16_c)
    # 17: commentary
    # 18: commentary

    # Questions 19-20
    # 19: commentary
    log_answer("q20", q20)

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return f"{UNFINISHED} unfinished questions"


if __name__ == "__main__":
    log_answer("PART 1", PART_1_PIPELINE)

"""
=== END OF PART 1 ===
"""
