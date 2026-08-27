# Controller

You interact with the environment through coding **controllers**, i.e. classes inheriting
`BaseController` (from `code_library/base_controller.py`) that implement the
`act(self, obs) -> action` method. You can do that in your code library; see for example
`code_library/example_controller.py`.

You analyse the environment through coding and executing **interactive scripts**, i.e.
scripts that instantiate the environment, run a controller on it, and operate on the
observations to print/log/compute information. You can use the provided
`code_library/interactive_script_example.py` as a template, which runs the example dummy
controller on the environment and prints some part of the observations.

We advise you follow this approach to solve the game:
- 1.a) When needing to create/edit a controller policy, do it as a controller class in `code_library/`.
- 1.b) When needing to create/edit an exploration script, do it in `code_library/`.
- 2) When needing to run a controller in an exploration script, edit the script to import
  the controller, and execute the script.
- 3) After a script result, observe its output to understand the environment, and iterate on
  (1) on the controllers/scripts. Balance exploration and exploitation.
- 4) Once you have a performing controller, submit it by importing it into `solution.py`'s
  `get_controller` function and running `python framework/control.py SubmitSolution`.

`SubmitSolution` and `ExitTask` are NOT callable tools - they are run from your working directory
as `python framework/control.py <ToolName>`. When you have finished - your best controller is in
`solution.py` and you do not wish to improve it further - end the run with
`python framework/control.py ExitTask`.
