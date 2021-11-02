import inspect

from typing import List, Callable, Dict, Any


def get_parameters(func: Callable, *args: Dict[str, Any]):
    signature = inspect.signature(func)
    func_parameters = list(signature.parameters.keys())

    # instance the parameters with the default parameters
    parameters = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    # update the parameters based on the ones sent by the user, take into consideration the order
    parameters.update({
        parameter: user_parameters[parameter]
        for parameter in func_parameters
        for user_parameters in args[::-1]
        if parameter in user_parameters
    })
    return parameters
