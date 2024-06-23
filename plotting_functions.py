import numpy as np
import plotly.graph_objects as go
from helpers import *
import pandas as pd


def plot_iterations(iterations, objective_values, max_frames=50):
    # Reducing the number of frames for animation
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=iterations,
                             y=objective_values,
                             mode='lines',
                             line=dict(color='blue', width=1)
                             )
                  )

    # Initialize frames list to store each frame
    frames = []

    # Determine frame step size
    frame_step = max(len(objective_values) // max_frames, 1)

    # Create frames for animation
    for i in range(0, len(objective_values), frame_step):
        frame = go.Frame(
            data=[go.Scatter(x=iterations[:i + 1],
                             y=objective_values[:i + 1],
                             mode='lines',
                             line=dict(color='blue', width=1)
                             )],
            name=f"frame_{i + 1}"
        )
        frames.append(frame)

    # Add frames to the figure
    fig.frames = frames

    # Update layout with animation buttons
    fig.update_layout(
        xaxis_title=r"$\text{Iteration Number}$",
        yaxis_title=r"$\text{Objective Function Value}$",
        xaxis_tickformat='d',  # Display integers on the x-axis
        yaxis_tickformat='.2e',  # Scientific notation for y-axis ticks
        yaxis_showexponent='all',  # Show exponent on y-axis ticks
        font=dict(family='Arial', size=14),
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": False}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        plot_bgcolor='white',  # Set plot background color
        hovermode='x',  # Show hover information only for x-axis
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'  # Adjust grid color
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'  # Adjust grid color
        )
    )

    fig.show()
    fig.write_html('animated_plot.html')


def plot_frequencies(frequencies_hz):
    # Create the bar plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f'Mode {i+1}' for i in range(len(frequencies_hz))],  # Adjusted to start from Mode 1
        y=frequencies_hz,
        marker=dict(color='red', line=dict(width=0)),  # Set bar color to red and remove bar outline
        width=0.1,  # Set the width of the bars
    ))

    # Annotate each bar with its frequency value using LaTeX formatting
    for i, freq in enumerate(frequencies_hz):
        fig.add_annotation(
            x=i,  # X coordinate of the annotation (mode number)
            y=freq,  # Y coordinate of the annotation (frequency value)
            text=f"${freq:.3f}$ Hz",  # Text of the annotation with 3 decimal places in LaTeX format
            font=dict(color='black', size=12),  # Annotation text font
            showarrow=False,  # Do not show arrows
            align='center',  # Text alignment
            yshift=10  # Adjust vertical position
        )

    # Add a horizontal line at f=0
    fig.add_shape(
        type="line",
        x0=-0.5,  # Starting x-position (before the first bar)
        y0=0,
        x1=len(frequencies_hz) - 0.5,  # Ending x-position (after the last bar)
        y1=0,
        line=dict(
            color="black",
            width=1
        )
    )

    fig.update_layout(
        xaxis_title=r"$\text{Mode Number}$",  # LaTeX formatted x-axis title
        yaxis_title=r"$\text{Frequency (Hz)}$",  # LaTeX formatted y-axis title
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(frequencies_hz))),
            ticktext=[f"${'Mode ' + str(i+1)}$" for i in range(len(frequencies_hz))]  # LaTeX formatted tick labels
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.2)',
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,  # Gap between bars
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')  # Adjust y-axis grid

    fig.show()


def areas_to_excel(members, optimized_A, excel_filename):
    # Create a DataFrame with Member Index starting from 1 and Optimized Area values
    df_optimized_areas = pd.DataFrame({
        'Member Index': np.arange(len(members)) + 1,  # Adjust index to start from 1
        'Optimized Area (m^2)': optimized_A
    })

    # Write the DataFrame to Excel
    df_optimized_areas.to_excel(excel_filename, index=False)

    print(f"Data has been written to '{excel_filename}'.")