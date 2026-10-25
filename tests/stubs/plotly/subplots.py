from .graph_objects import Figure


def make_subplots(rows=1, cols=1, shared_xaxes=False, shared_yaxes=False, start_cell="top-left", print_grid=False,
                  horizontal_spacing=None, vertical_spacing=None, subplot_titles=None, column_widths=None,
                  row_heights=None, specs=None, insets=None, column_titles=None, row_titles=None,
                  x_title=None, y_title=None, figure=None):
    if row_heights is not None and len(row_heights) != rows:
        raise ValueError("row_heights must have one entry per row")
    fig = Figure()
    fig.rows = rows
    return fig
