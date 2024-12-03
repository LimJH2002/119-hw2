# ECS 119 Homework 2: Parallel and Distributed Pipelines

**Due date: Dec 9, 11:59pm**

This homework will explore building parallel data pipelines in PySpark.
There are 3 parts:

- Part 1: Released Monday, November 25.

- Parts 2 and 3: Released Tuesday, November 26.

## Getting Started

Clone this repository to your own machine (or open up a Codespace).
Important: please do not create a "fork"! If you work on your
own copy, it should be a private repository.
Please see instructions here: https://piazza.com/class/m12ef423uj5p5/post/87

Then open up and complete `part1.py`, `part2.py`, and `part3.py`.
You will need a working installation of PySpark.
For Windows users, we recommend GitHub Codespaces.

If you get stuck, try taking a look at the file hints.md.

If you are still stuck, please ask a question on Piazza!

## Submitting Your Code

To submit your homework, upload (via git repository or a zip file) to Gradescope, as in HW1.

## Grading Notes

In order to receive credit for your work, please follow the following guidelines:

- Make sure that `python3 part1.py`, `python3 part2.py`, and `python3 part3.py` run successfully with no errors, and the same for
`pytest part1.py`, `pytest part2.py`, and `pytest part3.py`.
We cannot give credit to code that doesn't run on our setup!

- Make sure that your `output/part1-answers.txt` is generated and up-to-date, and all ohter output files in the `output/` folder. These files should be included in your code that you commit/upload, and they should also be regenerated whenever you run the code.

- To double check, try downloading the code from your submission and running it, to make sure you didn't miss anything.

- Don't use hardcoded filepaths like `/Home/Users/YourName/Downloads/myfile.py` or other issues that wouldn't translate correctly when running on someone else's machine.

- If you are using GitHub to submit, make sure that you `git commit` and `git push` your latest code to your personal repository.

- Don't rename any functions or methods or change the function signatures unless asked to do so.

- As discussed in the syllabus, a small number of points on each homework (at most 10% of the grade) are reserved for style points.
This also includes whether your free response answers are present and thoughtful.
Here are some things to consider: are your variable names chosen appropriately? Have you added comments with `#` or docstrings with `"""` where appropriate? Have you removed any obsolete, unused code blocks, functions, or variables?
(This includes removing `#TODO` and `raise NotImplementedError`!)
