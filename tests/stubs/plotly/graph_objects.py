"""Minimal stand-in for plotly.graph_objects that rejects unknown properties and mismatched arrays."""
import numpy as np


def _n(v):
    return None if v is None else len(list(v))


class _Trace:
    ALLOWED: set = set()

    def __init__(self, **props):
        bad = set(props) - self.ALLOWED
        if bad:
            raise TypeError(f"{type(self).__name__}: invalid properties {sorted(bad)}")
        self.props = props

    def validate(self):
        pass


class Scatter(_Trace):
    ALLOWED = {"x", "y", "name", "mode", "line", "fill", "showlegend", "hovertemplate"}

    def validate(self):
        if _n(self.props.get("x")) != _n(self.props.get("y")):
            raise ValueError(f"Scatter {self.props.get('name')}: x/y lengths differ")


class Candlestick(_Trace):
    ALLOWED = {"x", "open", "high", "low", "close", "name", "increasing_line_color", "increasing_fillcolor",
               "decreasing_line_color", "decreasing_fillcolor", "showlegend"}

    def validate(self):
        lengths = {_n(self.props.get(k)) for k in ("x", "open", "high", "low", "close")}
        if len(lengths) != 1:
            raise ValueError(f"Candlestick: array lengths differ {lengths}")


class Bar(_Trace):
    ALLOWED = {"x", "y", "name", "marker_color", "opacity", "showlegend", "orientation", "hovertemplate"}

    def validate(self):
        nx, ny = _n(self.props.get("x")), _n(self.props.get("y"))
        if nx != ny:
            raise ValueError(f"Bar: x/y lengths differ ({nx} vs {ny})")
        mc = self.props.get("marker_color")
        if isinstance(mc, (list, tuple, np.ndarray)) and len(mc) != nx:
            raise ValueError("Bar: marker_color list length differs from data")


class Heatmap(_Trace):
    ALLOWED = {"z", "x", "y", "text", "texttemplate", "colorscale", "zmid", "showscale", "textfont", "hovertemplate"}

    def validate(self):
        z = np.asarray(self.props["z"], dtype=float)
        if z.shape != (len(self.props["y"]), len(self.props["x"])):
            raise ValueError(f"Heatmap: z shape {z.shape} doesn't match y/x")
        if np.asarray(self.props["text"]).shape != z.shape:
            raise ValueError("Heatmap: text shape differs from z")


class Figure:
    LAYOUT = {"height", "margin", "hovermode", "legend", "xaxis_rangeslider_visible", "bargap", "showlegend", "barmode",
              "font", "hoverlabel"}
    AXIS = {"title_text", "type", "side", "autorange", "range", "showgrid", "ticks", "ticklen", "hoverformat",
            "showspikes", "spikemode", "spikesnap", "spikethickness", "spikedash", "spikecolor", "gridcolor", "zeroline"}

    def __init__(self, data=None, layout=None):
        self.traces, self.layout, self.rows = [], {}, 1
        if data is not None:
            for trace in (data if isinstance(data, (list, tuple)) else [data]):
                self.add_trace(trace)

    def add_trace(self, trace, row=None, col=None):
        if not isinstance(trace, _Trace):
            raise TypeError("add_trace expects a trace")
        if row is not None and not 1 <= row <= self.rows:
            raise ValueError(f"row {row} outside a {self.rows}-row figure")
        trace.validate()
        self.traces.append((trace, row, col))
        return self

    def update_layout(self, **kw):
        bad = set(kw) - self.LAYOUT
        if bad:
            raise TypeError(f"update_layout: invalid properties {sorted(bad)}")
        self.layout.update(kw)
        return self

    def _axes(self, row, col, kw):
        bad = set(kw) - self.AXIS
        if bad:
            raise TypeError(f"update axes: invalid properties {sorted(bad)}")
        if row is not None and not 1 <= row <= self.rows:
            raise ValueError(f"axis row {row} outside a {self.rows}-row figure")
        return self

    def update_xaxes(self, patch=None, selector=None, overwrite=False, row=None, col=None, **kw):
        return self._axes(row, col, kw)

    def update_yaxes(self, patch=None, selector=None, overwrite=False, row=None, col=None, **kw):
        return self._axes(row, col, kw)

    def add_hline(self, y, *, line_dash=None, line_color=None, line_width=None, opacity=None, row=None, col=None,
                  annotation_text=None):
        if row is not None and not 1 <= row <= self.rows:
            raise ValueError(f"hline row {row} outside a {self.rows}-row figure")
        self.hlines = getattr(self, "hlines", []) + [(y, row)]
        return self

    def validate(self):
        for trace, _, _ in self.traces:
            trace.validate()
