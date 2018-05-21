# Generative Adversarial Networks Example
The contained project is an example generative adversarial networks. You can run it with `python gan.py`.

## Setup
`pipenv` will make running the project much easier:
>pipenv install

>pipenv shell

>python gan.py

Will install dependencies and and configure the environemnt in an isolated virtual python environment, so system packages/etc are not affected by it's dependencies.

`pipenv clean` will get rid of the [dependency] files when you no longer need them.

On OS X, if matplotlib is complaining about not being a System Framework, add
`backend: TkAgg` to your `~/.matplotlib/matplotlibrc`. (Create the shell config for matplotlib if you don't have it).
This uses a unviersal rendering backend instead of the system one, sidestepping the issue.