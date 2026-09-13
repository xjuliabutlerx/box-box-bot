"""Pure chart/table-building functions for the visuals side-channel
(agent/visuals.py) - deliberately Streamlit-free so these are unit-
testable like any other layer in this codebase, unlike streamlit_app.py
itself, which isn't covered by tests/ (UI is verified manually).
"""

import pandas as pd
import plotly.graph_objects as go

COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFF200",
    "HARD": "#F0F0F0",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
}
UNKNOWN_COMPOUND_COLOR = "#999999"


def build_table(data: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(data)


def build_tire_strategy_figure(data: list[dict]) -> go.Figure:
    df = pd.DataFrame(data).sort_values(["Driver", "Stint"])
    df["StintStart"] = df.groupby("Driver")["StintLength"].cumsum() - df["StintLength"]
    drivers = df["Driver"].unique().tolist()

    fig = go.Figure()
    for compound in df["Compound"].unique():
        subset = df[df["Compound"] == compound]
        fig.add_trace(
            go.Bar(
                y=subset["Driver"],
                x=subset["StintLength"],
                base=subset["StintStart"],
                orientation="h",
                name=str(compound),
                marker_color=COMPOUND_COLORS.get(str(compound).upper(), UNKNOWN_COMPOUND_COLOR),
                hovertemplate="%{y}: %{x} laps<extra>" + str(compound) + "</extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Tire Strategy",
        xaxis_title="Lap",
        yaxis_title="Driver",
        yaxis=dict(categoryorder="array", categoryarray=drivers[::-1]),
    )
    return fig


def build_track_map_figure(data: dict) -> go.Figure:
    points = data["points"]
    xs = [p["X"] for p in points]
    ys = [p["Y"] for p in points]
    speeds = [p["Speed"] for p in points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+lines",
            line=dict(width=1, color="rgba(0,0,0,0.15)"),
            marker=dict(size=6, color=speeds, colorscale="Turbo", colorbar=dict(title="km/h")),
            hovertemplate="Speed: %{marker.color:.0f} km/h<extra></extra>",
            showlegend=False,
        )
    )

    for corner in data.get("corners", []):
        fig.add_annotation(
            x=corner["X"],
            y=corner["Y"],
            text=str(corner["Number"]),
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="white",
        )

    title = f"{data.get('circuit', '')} — {data.get('driver', '')} ({data.get('lap_time', '')})"
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
    )
    return fig
