import inspect

class MaraTool(object):

    _user_params = { }

    @classmethod
    def update_user_params(self, user_params):
        self._user_params.update(user_params)

    @classmethod
    def populate_parser(self, method, parser):
        f = getattr(self, method)
        g = getattr(f, '_wrapped_method', f)
        args, varargs, varkw, defaults = inspect.getargspec(g)

        if defaults is None: defaults = [ ]

        for arg, default in zip(args[1:], defaults):
            if type(default) is list:
                parser.add_argument("--"+arg, action='append',
                                    type=type(default[0]))
            elif type(default) is bool:
                parser.add_argument("--"+arg, action='store_true')
            else:
                parser.add_argument("--"+arg, type=type(default),
                                    default=default)

        class kwarg_getter(object):
            def __init__(self, A):
                self.my_args = A
            def __call__(self, pargs):
                return {k:v for k,v in vars(pargs).iteritems()
                        if k in self.my_args}
        return kwarg_getter(args)

    def get_user_param(self, key, default=None):
         cls_params = self._user_params.get(self.__class__.__name__, None)
         return cls_params.get(key, default) if cls_params else default

    def set_plot_axes(self, ax):
        self._plot_axes = ax

    def get_plot_axes(self, **fig_kwds):
        ax0 = getattr(self, '_plot_axes', None)
        if ax0 is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(**fig_kwds)
            ax0 = fig.add_subplot(111)
        return ax0

    def set_show_action(self, action, callback=None, hardcopy=None):
        if callback: self._show_callback = callback
        if hardcopy: self._show_hardcopy = hardcopy
        self._show_action = action

    def show(self):
        if getattr(self, '_show_hardcopy', False):
            plt.savefig(self._show_hardcopy)
        elif getattr(self, '_show_callback', False):
            self._show_callback()
        elif getattr(self, '_show_action', False) == 'hold':
            return
        else:
            import matplotlib.pyplot as plt
            plt.show()
