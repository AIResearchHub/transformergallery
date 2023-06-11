



class Unlimiformer:

    def __init__(self, model):

        self.original_non_injected_decoder_layer_forward_funcs = []
        self.hook_handles = []

    def break_into(self, model):

        self.hooks_injected = False
        self.training_hooks_injected = False

        # Activate AttentionKNN when calling model.eval(), deactivate for model.train()
        self.original_model_eval_func = model.eval
        model.eval = self.pre_eval_hook
        self.original_model_train_func = model.train
        model.train = self.pre_train_hook
    def pre_eval_hook(self):
        """
        remove training hooks, inject eval hooks, and call original model.eval()
        """
        self.remove_training_hooks(self.model)
        self.inject_hooks(self.model)
        self.original_model_eval_func()

    def pre_train_hook(self, mode=True):
        """
        empty cache, if training, call self.break_out, inject training hooks and
        call original model.train()
        """
        torch.cuda.empty_cache()
        if mode is True:
            self.break_out(self.model)
            if self.unlimiformer_training:
                self.inject_training_hooks(self.model)
        self.original_model_train_func(mode)

    def inject_hooks(self, model):
        """inject hooks for eval"""
        if self.hooks_injected:
            return

        self.hooks_injected = True

    def inject_training_hooks(self, model):
        """inject hooks for train"""
        if self.training_hooks_injected:
            return

        model.forward = self.pre_forward_hook





        self.training_hooks_injected = True

    def inject_hooks_for_unaffected_layers(self, model, decoder_layers_to_run):
        """inject hooks for unaffected layers"""
        self.original_non_injected_decoder_layer_forward_funcs = []
        non_injected_decoder_layers = [l for l in self.attention_layer_to_run(0, None)
            if l not in decoder_layers_to_run]
        for decoder_layer in non_injected_decoder_layers:
            self.original_non_injected_decoder_layer_forward_funcs.append(decoder_layer.forward)
            decoder_layer.forward = self.create_noninjected_decoder_layer_func(decoder_layer.forward, decoder_layer)

    def create_self_attn_pre_forward_hook(self, original_self_attn_forward_func):
        """
        modify original self attention forward function to not take
        in any past key and values
        """
        def self_attention_pre_forward_hook(*args, **kwargs):
            kwargs['past_key_value'] = None
            return original_self_attn_forward_func(*args, **kwargs)

        return self_attention_pre_forward_hook

    def create_decoder_layer_func(self, decoder_layer_original_forward_func, decoder_layer):
        pass

    def create_noninjected_decoder_layer_func(self, decoder_layer_original_forward_func, decoder_layer):
        pass

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self, model):
        """
        Remove all the hook handles and change all the model functions such as
        model.generate, model.forward, model._reorder_cache and
        decoder layer cross attention forward to its original function
        """
        torch.cuda.empty_cache()

        if not self.hooks_injected:
            return

        for h in self.hook_handles:
            h.remove()

        model.generate = self.original_generate_func
        model.forward = self.original_forward_func
        model._reorder_cache = self.original_reorder_cache_func

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            self.cross_attention(decoder_layer).forward = original_func

        self.hooks_injected = False

    def remove_training_hooks(self, model):
        """
        change model.forward back to its original function, for all the decoder layers,
        change self attention forward, cross attention forward, and forward all back
        to its original function. For all the non injected forward functions,
        change the layer.forward back to its original function
        """
        if not self.training_hooks_injected:
            return

        for h in self.hook_handles:
            h.remove()
        model.forward = self.original_forward_func

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_self_attn_forward_funcs):
            self.self_attention(decoder_layer).forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            self.cross_attention(decoder_layer).forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func

        non_injected_decoder_layers = [l for l in self.attention_layer_to_run(0, None)
            if l not in decoder_layers_to_run]
        for decoder_layer, original_func in zip(non_injected_decoder_layers, self.original_non_injected_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func

        self.training_hooks_injected = False

    def reset_memory(self, input_ids, attention_mask):
        pass

    def chunked_encode_input(self, input_ids, attention_mask):
        pass

    def window_indices(self, total_seq_len):
        pass

    def pre_generate_hook(self, input_ids, **kwargs):
        pass

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        A function that preprocesses the inputs for the original forward function
        and

        not sure why set gradient checkpointing to false
        """

        self.model.base_model.decoder.gradient_checkpointing = False

        if not self.is_input_encoding_pass:
            if self.model.training:
                self.long_inputs_encoded, self.long_inputs_mask = self.chunked_encode_input(input_ids=input_ids, attention_mask=attention_mask)
                input_ids = input_ids[:, :self.actual_model_window_size]
                attention_mask = attention_mask[:, :self.actual_model_window_size] if attention_mask is not None else None
            else:
                if kwargs.get('past_key_values') is None:
                    self.is_first_test_decoding_step = True

                if input_ids is not None:
                    self.input_ids = torch.cat([self.input_ids, input_ids[0]])
                if kwargs.get('decoder_input_ids') is not None:
                    self.generated_input_ids = torch.cat([self.generated_input_ids, kwargs['decoder_input_ids']], axis=-1)

        result = self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
        self.is_first_test_decoding_step = False
        return result

    def create_cross_attn_pre_forward_hook(self, original_cross_attn_forward_func, decoder_layer, i):
        pass

    def attention_forward_hook(self, module, input, output):
        pass

    def train_attention_forward_hook(self, module, input, output):
        pass

    def reorder_cache_hook(self, past, beam_idx):
        pass

