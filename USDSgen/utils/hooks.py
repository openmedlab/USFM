class SaveOutput:
    def __init__(self, name=None):
        self.module = []
        self.module_in = []
        self.module_out = []
        self.name = name

    def __call__(self, module, module_in, module_out):
        self.module.append(module)
        self.module_in.append(module_in)
        self.module_out.append(module_out)

    def clear(self):
        self.module = []
        self.module_in = []
        self.module_out = []


class LayerForwardHooks:
    def __init__(self):
        self.hooks = {}

    def register(self, model, layer_names):
        self.hook_handles = []
        for name, m in model.named_modules():
            if name in layer_names:
                self.hooks[name] = SaveOutput(name)
                handle = m.register_forward_hook(self.hooks[name])
                self.hook_handles.append(handle)

    def clear_hooks(self, layer_names):
        for name in layer_names:
            self.hooks[name].clear()

    def reset_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

        self.hook_handles = []
        self.hooks = {}
