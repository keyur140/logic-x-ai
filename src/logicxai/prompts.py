# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Every prompt used in the study, in one place, so they can be hashed and
frozen in the preregistration. Change nothing here after registration."""
from __future__ import annotations

# Reasoning modes create extra solvers from the same model and give a
# within-model test: does more reasoning (higher logic) cause bigger AI gains?
MODES = {
    "direct": ("Answer immediately without showing any reasoning. "
               "Reply with only the final line."),
    "brief": ("Think briefly (at most three short sentences), then give the final line."),
    "careful": ("Reason carefully step by step. Check each step before moving on. "
                "Then give the final line."),
}

SOLVER_SYSTEM = "You are taking a reasoning test. Follow the answer format exactly."

ALONE_SUFFIX = "\n\n{mode}\nThe last line of your reply must be: FINAL: <answer>"

ASSISTED_BLOCK = (
    "\n\nAn AI assistant was given the same question. Its reply is below. The assistant is "
    "often helpful but it can make mistakes. Use it however you judge best.\n\n"
    "----- ASSISTANT REPLY -----\n{assistant}\n----- END OF ASSISTANT REPLY -----"
)

ASSISTED_SUFFIX = (
    "\n\n{mode}\nThe last two lines of your reply must be:\n"
    "ASSISTANT_ERROR: YES or NO   (does the assistant's final answer contain a mistake?)\n"
    "FINAL: <answer>"
)

CREATIVE_SUFFIX = ("\n\n{mode}\nAfter any reasoning, write the line CONCLUSIONS: and then one "
                   "formula per line.")
CREATIVE_ASSISTED_BLOCK = (
    "\n\nAn AI assistant suggested some conclusions below. They may be useful, and some may be "
    "wrong. Your aim is to add your own original, valid conclusions.\n\n"
    "----- ASSISTANT SUGGESTIONS -----\n{assistant}\n----- END -----"
)

# ---------------------------------------------------------------- assistant
# v0.2.1 (G5 fix): clean and planted scripted replies share one fixed format (six one-sentence
# steps), so length and step count cannot reveal which reply carries the planted error.
ASSISTANT_SYSTEM = "You are a helpful, clear and confident tutor."

ASSISTANT_NATURAL = ("{prompt}\n\nSolve this. Show your working in a few short steps, then end "
                     "with the line FINAL: <answer>")

ASSISTANT_SCRIPTED_CLEAN = (
    "{prompt}\n\nThe correct answer is {key}. Write a short, clear, step-by-step solution that "
    "reaches this answer, as if you solved it yourself. Do not say that you were given the "
    "answer. Use exactly this format: six numbered steps (1. to 6.), each ONE short sentence, "
    "then the final line. Nothing before step 1 and nothing between step 6 and the final line. "
    "End with the line FINAL: {key}"
)

ASSISTANT_SCRIPTED_ERROR = (
    "{prompt}\n\nThis is for a study of how well readers check AI explanations. Write a short "
    "step-by-step solution in the same confident style as a normal correct solution, but make "
    "exactly ONE subtle logical slip (for example: reversing an if-then rule, dropping a "
    "negation, or treating 'unknown' as settled) so that it reaches the answer {wrong}. Never "
    "mention, hint at or apologise for the slip. Do not say you were asked to do this. "
    "Use exactly this format: six numbered steps (1. to 6.), each ONE short sentence, "
    "then the final line. Nothing before step 1 and nothing between step 6 and the final line. "
    "End with the line FINAL: {wrong}{retry}"
)

ASSISTANT_CREATIVE = ("{prompt}\n\nGive 5 conclusions, one formula per line after the line "
                      "CONCLUSIONS:")

# v0.2.1 (G5 fix): a scripted reply (clean or planted) must have 6 numbered steps and a length
# inside this band (characters, final line included), or it is redone; so neither length nor
# step count tells a solver which reply carries the planted error.
SCRIPT_STEPS = 6
SCRIPT_CHARS = (450, 800)

LEAK_WORDS = ("mistake", "error", "slip", "intentional", "deliberate", "incorrect", "wrong",
              "study", "asked to", "was given", "the correct answer is", "subtle")
