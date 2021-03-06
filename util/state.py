#!/usr/bin/env python


class DecoderState(object):

    def __init__(self, hidden=None, **kwargs):

        if hidden is not None:
            self.hidden = hidden
        for k, v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def get_batch_size(self):
        if self.hidden is not None:
            return self.hidden.size(1)
        else:
            return next(iter(self.__dict__.values())).size(1)

    def size(self):
        sizes = {k: v.size() for k, v in self.__dict__.items()}
        return sizes

    def slice_select(self, stop):
        kwargs = {}
        for k, v in self.__dict__.items():
            if "hidden" in k:
                kwargs[k] = v[:, :stop].clone()
            else:
                kwargs[k] = v[:stop]
        return DecoderState(**kwargs)

    def index_select(self, indices):
        kwargs = {}
        for k, v in self.__dict__.items():
            if 'hidden' in k:
                kwargs[k] = v.index_select(1, indices)
            else:
                kwargs[k] = v.index_select(0, indices)
        return DecoderState(**kwargs)

    def mask_select(self, mask):
        kwargs = {}
        for k, v in self.__dict__.items():
            if "hidden" in k:
                kwargs[k] = v[:, mask]
            else:
                kwargs[k] = v[mask]
        return DecoderState(**kwargs)

    def _inflate_tensor(self, X, times):
        sizes = X.size()

        if X.dim() == 1:
            X = X.unsqueeze(1)

        repeat_times = [1] * X.dim()
        repeat_times[1] = times
        X = X.repeat(*repeat_times).view(-1, *sizes[1:])
        return X

    def inflate(self, times):
        """
        inflate
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if "hidden" in k:
                num_layers, batch_size, _ = v.size()
                kwargs[k] = v.repeat(1, 1, times).view(
                    num_layers, batch_size * times, -1)
            else:
                kwargs[k] = self._inflate_tensor(v, times)
        return DecoderState(**kwargs)
