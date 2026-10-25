"""Minimal stand-in for Streamlit used to exercise app.py without a browser.

Every function mirrors the real Streamlit signature for the parameters the app uses, so a
wrong keyword raises TypeError. Widgets return defaults unless a test overrides them.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import numbers

import pandas as _pd

from . import column_config  # noqa: F401  (exposed as st.column_config)


class StopExecution(Exception):
    pass


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


session_state = SessionState()


class _State:
    def __init__(self):
        self.overrides: dict = {}
        self.calls: list = []
        self.keys: set = set()
        self.issues: list = []
        self.callbacks: dict = {}

    def reset_run(self):
        self.calls, self.keys, self.issues, self.callbacks = [], set(), [], {}


STATE = _State()


def _log(kind, **data):
    STATE.calls.append((kind, data))


def _issue(msg):
    STATE.issues.append(msg)


def _register(key, kind):
    if key is None:
        return
    if key in STATE.keys:
        raise RuntimeError(f"Duplicate key {key!r} ({kind}) - Streamlit would raise DuplicateElementKey")
    STATE.keys.add(key)


def _value(key, label, default):
    if key is not None and key in STATE.overrides:
        val = STATE.overrides[key]
    elif label in STATE.overrides:
        val = STATE.overrides[label]
    elif key is not None and key in session_state:
        val = session_state[key]
    else:
        val = default
    if key is not None:
        session_state[key] = val
    return val


class _Container:
    def __init__(self, name):
        self._name = name

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, item):
        target = globals().get(item)
        if callable(target):
            return target
        raise AttributeError(f"Streamlit container has no attribute {item!r}")


sidebar = _Container("sidebar")


# --- page & text ------------------------------------------------------------
def set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None):
    _log("set_page_config", page_title=page_title, page_icon=page_icon, layout=layout)


def _text(kind, body):
    if not isinstance(body, str):
        _issue(f"{kind} body is not a string: {type(body)}")
    _log(kind, body=str(body))


def title(body, anchor=None, *, help=None, width="stretch"):
    _text("title", body)


def header(body, anchor=None, *, help=None, divider=False, width="stretch"):
    _text("header", body)


def subheader(body, anchor=None, *, help=None, divider=False, width="stretch"):
    _text("subheader", body)


def markdown(body, unsafe_allow_html=False, *, help=None, width="stretch"):
    _text("markdown", body)


def caption(body, unsafe_allow_html=False, *, help=None, width="stretch"):
    _text("caption", body)


def write(*args, unsafe_allow_html=False):
    _log("write", args=args)


def divider(*, width="stretch"):
    _log("divider")


def info(body, *, icon=None, width="stretch"):
    _text("info", body)


def warning(body, *, icon=None, width="stretch"):
    _text("warning", body)


def error(body, *, icon=None, width="stretch"):
    _text("error", body)


def success(body, *, icon=None, width="stretch"):
    _text("success", body)


def metric(label, value, delta=None, delta_color="normal", *, help=None, label_visibility="visible", border=False,
           width="stretch", height="content", chart_data=None, chart_type="line", delta_arrow="auto"):
    for name, v in (("value", value), ("delta", delta)):
        if v is not None and not isinstance(v, (str, int, float)):
            _issue(f"metric {label!r} {name} has unsupported type {type(v)}")
    _log("metric", label=label, value=value, delta=delta)


# --- data & charts ------------------------------------------------------------
def _check_frame(df, column_config_, where):
    if not isinstance(df, _pd.DataFrame):
        _issue(f"{where}: data is {type(df)}")
        return
    if df.columns.duplicated().any():
        _issue(f"{where}: duplicate column names {list(df.columns[df.columns.duplicated()])}")
    for c in df.columns:
        if not isinstance(c, str):
            _issue(f"{where}: non-string column name {c!r} ({type(c).__name__})")
    for c in df.columns:
        col = df[c]
        if col.dtype == object:
            kinds = {type(v).__name__ for v in col.dropna()}
            if len(kinds) > 1 and not kinds <= {"float", "int"}:
                _issue(f"{where}: mixed types in column {c!r}: {sorted(kinds)}")
    if column_config_:
        for k in column_config_:
            if k not in df.columns and k != "_index":
                _issue(f"{where}: column_config refers to missing column {k!r}")


def dataframe(data=None, width="stretch", height="auto", *, use_container_width=None, hide_index=None,
              column_order=None, column_config=None, key=None, on_select="ignore", selection_mode="multi-row",
              row_height=None, placeholder=None):
    if use_container_width is not None:
        _issue("st.dataframe: use_container_width is deprecated")
    if not (height == "auto" or isinstance(height, int)):
        _issue(f"st.dataframe: bad height {height!r}")
    _register(key, "dataframe")
    _check_frame(data, column_config, f"dataframe[{key}]")
    _log("dataframe", key=key, shape=getattr(data, "shape", None), data=data)


def plotly_chart(figure_or_data, use_container_width=None, *, width="stretch", height="content", theme="streamlit",
                 key=None, on_select="ignore", selection_mode=("points", "box", "lasso"), config=None, **kwargs):
    if kwargs:
        _issue(f"st.plotly_chart: deprecated **kwargs {sorted(kwargs)}")
    if use_container_width is not None:
        _issue("st.plotly_chart: use_container_width is deprecated")
    _register(key, "plotly_chart")
    validate = getattr(figure_or_data, "validate", None)
    if validate:
        validate()
    _log("plotly_chart", key=key, figure=figure_or_data)


def download_button(label, data, file_name=None, mime=None, key=None, help=None, on_click="rerun", args=None,
                    kwargs=None, *, type="secondary", icon=None, disabled=False, use_container_width=None,
                    width="content"):
    _register(key, "download_button")
    if not isinstance(data, (bytes, str)):
        _issue(f"download_button {label!r}: data is {type(data)}")
    if not file_name:
        _issue(f"download_button {label!r}: missing file_name")
    _log("download_button", label=label, key=key, file_name=file_name, mime=mime, data=data)
    return bool(STATE.overrides.get(key, False))


def button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, type="secondary", icon=None,
           disabled=False, use_container_width=None, width="content"):
    _register(key, "button")
    if on_click is not None:
        STATE.callbacks[key] = (on_click, args or (), kwargs or {})
    _log("button", label=label, key=key)
    return bool(STATE.overrides.get(key, False))


# --- widgets ------------------------------------------------------------------
def text_input(label, value="", max_chars=None, key=None, type="default", help=None, autocomplete=None,
               on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False,
               label_visibility="visible", icon=None, width="stretch"):
    _register(key, "text_input")
    val = _value(key, label, value)
    _log("text_input", label=label, key=key, value=val)
    return val


def _choice(kind, label, options, index, format_func, key, accept_new=False):
    opts = list(options)
    _register(key, kind)
    if not opts and not accept_new:
        _issue(f"{kind} {label!r} has no options")
        return None
    if index is not None and not (0 <= index < len(opts)):
        raise ValueError(f"{kind} {label!r}: index {index} out of range")
    for o in opts:
        format_func(o)  # Streamlit calls it for every option
    val = _value(key, label, None if index is None else opts[index])
    if val is not None and val not in opts:
        if not accept_new:
            raise ValueError(f"{kind} {label!r}: value {val!r} not in options")
        format_func(val)
    _log(kind, label=label, key=key, value=val, options=opts, labels=[format_func(o) for o in opts])
    return val


def selectbox(label, options, index=0, format_func=str, key=None, help=None, on_change=None, args=None,
              kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible",
              accept_new_options=False, filter_mode="fuzzy", width="stretch", bind=None):
    if accept_new_options and filter_mode is None:
        raise ValueError("filter_mode=None is incompatible with accept_new_options=True")
    return _choice("selectbox", label, options, index, format_func, key, accept_new=accept_new_options)


def radio(label, options, index=0, format_func=str, key=None, help=None, on_change=None, args=None, kwargs=None,
          *, disabled=False, horizontal=False, captions=None, label_visibility="visible", width="content"):
    return _choice("radio", label, options, index, format_func, key)


def checkbox(label, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False,
             label_visibility="visible", width="content"):
    _register(key, "checkbox")
    val = _value(key, label, value)
    _log("checkbox", label=label, key=key, value=val)
    return bool(val)


def number_input(label, min_value=None, max_value=None, value="min", step=None, format=None, key=None, help=None,
                 on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False,
                 label_visibility="visible", icon=None, width="stretch"):
    _register(key, "number_input")
    numeric = {n: v for n, v in (("min_value", min_value), ("max_value", max_value), ("value", value),
                                 ("step", step)) if v is not None and v != "min"}
    for n, v in numeric.items():
        if isinstance(v, bool) or not isinstance(v, numbers.Number):
            raise TypeError(f"number_input {label!r}: {n} must be a number, got {type(v)}")
    kinds = {"int" if isinstance(v, numbers.Integral) else "float" for v in numeric.values()}
    if len(kinds) > 1:
        raise TypeError(f"number_input {label!r}: mixed int/float arguments {numeric} (StreamlitMixedNumericTypesError)")
    if value == "min":
        value = min_value if min_value is not None else 0.0
    if min_value is not None and value < min_value:
        raise ValueError(f"number_input {label!r}: value {value} below min {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"number_input {label!r}: value {value} above max {max_value}")
    if format is not None:
        format % value  # raises if the printf format doesn't suit the value
    val = _value(key, label, value)
    _log("number_input", label=label, key=key, value=val, help=help)
    return val


def date_input(label, value="today", min_value=None, max_value=None, key=None, help=None, on_change=None, args=None,
               kwargs=None, *, format="YYYY/MM/DD", disabled=False, label_visibility="visible", width="stretch"):
    _register(key, "date_input")
    if value == "today":
        value = _dt.date.today()
    if max_value is not None and value > max_value:
        raise ValueError(f"date_input {label!r}: value after max")
    if min_value is not None and value < min_value:
        raise ValueError(f"date_input {label!r}: value before min")
    val = _value(key, label, value)
    _log("date_input", label=label, key=key, value=val)
    return val


# --- layout -------------------------------------------------------------------
def columns(spec, *, gap="small", vertical_alignment="top", border=False, width="stretch"):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Container(f"column{i}") for i in range(n)]


def tabs(tabs, *, width="stretch", default=None):
    labels = list(tabs)
    if len(set(labels)) != len(labels):
        _issue(f"st.tabs: duplicate labels {labels}")
    _log("tabs", labels=labels)
    return [_Container(f"tab:{t}") for t in labels]


def container(*, border=None, key=None, height=None, width="stretch", horizontal=False,
              horizontal_alignment="left", vertical_alignment="top", gap="small"):
    return _Container("container")


def expander(label, expanded=False, *, icon=None, width="stretch"):
    _log("expander", label=label)
    return _Container(f"expander:{label}")


@contextlib.contextmanager
def spinner(text="In progress...", *, show_time=False, width="content"):
    yield


def stop():
    raise StopExecution()


def cache_data(func=None, *, ttl=None, max_entries=None, show_spinner=True, persist=None, hash_funcs=None,
               scope="global"):
    def deco(f):
        f.clear = lambda *a, **k: None
        return f
    return deco(func) if func is not None else deco


class RerunRequested(Exception):
    pass


def rerun(*, scope="app"):
    raise RerunRequested()


def toggle(label, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False,
           label_visibility="visible", width="content"):
    _register(key, "toggle")
    val = _value(key, label, value)
    _log("toggle", label=label, key=key, value=val)
    return bool(val)


def segmented_control(label, options, *, selection_mode="single", default=None, format_func=None, key=None, help=None,
                      on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible",
                      width="content", bind=None):
    opts = list(options)
    _register(key, "segmented_control")
    if key is not None and key in session_state and session_state[key] is not None and default is not None \
            and key not in STATE.overrides:
        _issue(f"segmented_control {key!r}: default passed while Session State already holds a value "
               "(Streamlit shows a warning)")
    if default is not None and default not in opts:
        raise ValueError(f"segmented_control {label!r}: default {default!r} not in options")
    if key is not None and key in STATE.overrides and STATE.overrides[key] is None:
        # simulate the user clicking the active option: Streamlit sets None, then runs on_change
        session_state[key] = None
        if on_change:
            on_change(*(args or ()), **(kwargs or {}))
        val = session_state[key]
        STATE.overrides.pop(key)
    else:
        val = _value(key, label, default)
    if val is not None and val not in opts:
        raise ValueError(f"segmented_control {label!r}: value {val!r} not in options {opts}")
    if key is not None:
        session_state[key] = val
    _log("segmented_control", label=label, key=key, value=val, options=opts)
    return val


def pills(label, options, *, selection_mode="single", default=None, format_func=None, key=None, help=None,
          on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible", width="content", bind=None):
    opts = list(options)
    _register(key, "pills")
    if key is not None and key in session_state and default is not None and key not in STATE.overrides:
        _issue(f"pills {key!r}: default passed while Session State already holds a value")
    if selection_mode == "multi":
        default = list(default or [])
        bad = [d for d in default if d not in opts]
        if bad:
            raise ValueError(f"pills {label!r}: default {bad} not in options")
    val = _value(key, label, default)
    if selection_mode == "multi" and any(v not in opts for v in (val or [])):
        raise ValueError(f"pills {label!r}: value {val!r} not in options")
    _log("pills", label=label, key=key, value=val, options=opts)
    return val


def multiselect(label, options, default=None, format_func=str, key=None, help=None, on_change=None, args=None,
                kwargs=None, *, max_selections=None, placeholder=None, disabled=False, label_visibility="visible",
                accept_new_options=False, width="stretch", bind=None):
    opts = list(options)
    _register(key, "multiselect")
    if key is not None and key in session_state and default not in (None, []) and key not in STATE.overrides:
        _issue(f"multiselect {key!r}: default passed while Session State already holds a value")
    default = list(default or [])
    bad = [d for d in default if d not in opts]
    if bad:
        raise ValueError(f"multiselect {label!r}: default {bad} not in options")
    for o in default:
        format_func(o)
    val = list(_value(key, label, default) or [])
    if not accept_new_options and any(v not in opts for v in val):
        raise ValueError(f"multiselect {label!r}: value not in options")
    if max_selections is not None and len(val) > max_selections:
        raise ValueError(f"multiselect {label!r}: more than {max_selections} selections")
    _log("multiselect", label=label, key=key, value=val, options_count=len(opts))
    return val


def data_editor(data, *, width="stretch", height="auto", use_container_width=None, hide_index=None, column_order=None,
                column_config=None, num_rows="fixed", disabled=False, key=None, on_change=None, args=None, kwargs=None,
                row_height=None, placeholder=None):
    _register(key, "data_editor")
    _check_frame(data, column_config, f"data_editor[{key}]")
    if not isinstance(disabled, bool):
        missing = [c for c in disabled if c not in data.columns]
        if missing:
            _issue(f"data_editor {key!r}: disabled columns {missing} don't exist")
    val = STATE.overrides.get(key, data)
    _log("data_editor", key=key, data=val)
    return val


import types as _types

context = _types.SimpleNamespace(theme=_types.SimpleNamespace(type="light", base="light"))


def fragment(func=None, *, run_every=None):
    def deco(f):
        f.__is_fragment__ = True
        return f
    return deco(func) if func is not None else deco


def toast(body, *, icon=None, duration="short"):
    _log("toast", body=str(body), icon=icon)


class _Empty(_Container):
    def __init__(self):
        super().__init__("empty")

    def container(self, **kwargs):
        return _Container("empty-container")

    def empty(self):
        _log("empty_cleared")


def empty():
    return _Empty()
